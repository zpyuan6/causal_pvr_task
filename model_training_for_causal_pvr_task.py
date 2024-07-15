import torch
import wandb
import os
from model.model_training import load_model, train_model, val_model
from data.CausalPVRDataset import CausalPVRDataset
from model.pytorchtools import EarlyStopping
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import tqdm

def load_dataset(dataset_path, dataset_name, batch_size, is_return_dataset=False, return_pure_img=False):
    train_dataset = CausalPVRDataset(dataset_path, dataset_name, "train", return_pure_img)
    val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val", return_pure_img)
    print(f"train dataset {len(train_dataset)}, val dataset {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

    if is_return_dataset:
        return train_dataloader, val_dataloader, train_dataset, val_dataset

    return train_dataloader, val_dataloader

def train_pvr(dataset_folder,dataset_name):
    batch_size = 128    
    num_class = 10
    learn_rate=0.001
    NUM_EPOCHES=500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login(key=os.getenv('WANDB_KEY'))
    

    models = ['vgg', 'resnet', 'densenet', 'mobilenet']

    for model_name in models:
        wandb.init(
            project="causal_pvr_model_training",
            name = f"{dataset_name}_{model_name}"
        )

        print("Start {} model training".format(model_name))

        model = load_model(model_name, num_class=num_class)
        
        model.to(device)

        train_dataloader, val_dataloader = load_dataset(dataset_folder, dataset_name, batch_size)

        model_save_path = os.path.join(dataset_folder,f"{model_name}.pt")
        early_stopping = EarlyStopping(patience=50, verbose=True, path=model_save_path)

        optimizer = optim.AdamW(model.parameters(), learn_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        loss_function = nn.CrossEntropyLoss()

        best_acc = 0
        
        for epoch in range(NUM_EPOCHES):
            train_loss, train_acc = train_model(model, loss_function, optimizer, device, NUM_EPOCHES, epoch, train_dataloader)
            avgloss, correct, acc = val_model(model, device, loss_function, val_dataloader)
            scheduler.step()

            early_stopping(avgloss, acc, model, train_loss)
            log = {f'training loss {model_name}': train_loss, 
                f'training acc {model_name}': train_acc,
                f'val loss {model_name}': avgloss, 
                f'val acc {model_name}': acc, 
                'epoch':epoch}
            wandb.log(log)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(dataset_folder, f"{model_name}_best.pt") )

            if early_stopping.early_stop:
                print("Early stopping")
                break

def test_pvr(dataset_folder,dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ['vgg', 'resnet', 'densenet', 'mobilenet']

    for model_name in models:
        model_parameter_path = os.path.join(dataset_folder,f"{model_name}.pt")

        model = load_model(model_name, model_parameter_path, num_class=10)
        # model = load_model("mobilenet", "model\logs\mobilenet.pt")
        model.eval()
        model.to(device)

        val_folder = "F:\\Broden\\opensurfaces"

        train_dataloader, val_dataloader = load_dataset(dataset_folder, dataset_name, 32)

    
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        test_recall = torchmetrics.Recall(task="multiclass",average='none', num_classes=10).to(device)
        test_precision = torchmetrics.Precision(task="multiclass",average='none', num_classes=10).to(device)
        test_auc = torchmetrics.AUROC(task="multiclass",average="macro", num_classes=10).to(device)

        total_num = len(val_dataloader.dataset)

        TP = torch.zeros(7)
        FP = torch.zeros(7)
        TN = torch.zeros(7)
        FN = torch.zeros(7)

        with torch.no_grad():
            with tqdm.tqdm(total = len(val_dataloader)) as pbar:
                for data, target, _ in val_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    test_acc(torch.argmax(output, 1), target)
                    test_auc.update(output, target)
                    test_recall(torch.argmax(output, 1), target)
                    test_precision(torch.argmax(output, 1), target)

                    pbar.update(1)

        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        total_auc = test_auc.compute()
        print(f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        print("recall of every test dataset class: ", total_recall, torch.sum(total_recall)/7)
        print("precision of every test dataset class: ", total_precision, torch.sum(total_precision)/7)
        print("auc:", total_auc.item())

        # 清空计算对象
        test_precision.reset()
        test_acc.reset()
        test_recall.reset()
        test_auc.reset()



if __name__ == "__main__":

    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\chain", dataset_name = "mnist")
    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_chain", dataset_name = "mnist")
    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\collider", dataset_name = "mnist")
    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_collider", dataset_name = "mnist")
    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\fork", dataset_name = "mnist")
    # train_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_fork", dataset_name = "mnist")

    test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\chain", dataset_name = "mnist")
    # test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_chain", dataset_name = "mnist")
    # test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\collider", dataset_name = "mnist")
    # test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_collider", dataset_name = "mnist")
    # test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\fork", dataset_name = "mnist")
    # test_pvr(dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr\\limited_fork", dataset_name = "mnist")
    

    