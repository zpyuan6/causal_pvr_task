from data.CausalPVRDataset import CausalPVRDataset
import random
from model.model_training import load_model
import torchvision.transforms as T
import torch
import os
from explanation.heatmapes.grad_cam import generate_grad_cam_from_img
from explanation.concept_based.cav import train_cav_for_pvr_task, show_concept_dataset, calculate_local_cav_sensitivity, calculate_global_tcav, identify_samples_based_on_cav
from explanation.concept_based.crp_relmax import conditional_attributions, feature_visualization, identify_graph
from explanation.causal.cexCNN import filter_importance, cexCNN_heatmap
from explanation.causal.concept_causal_map import train_cp_for_pvr_task, show_concept_dataset_for_cp, calculate_local_concept_sensitivity, concept_detection, calculate_local_concept_sensitivity_based_on_gradient, identify_global_concept_causality_graph
# calculate_global_concept_sensitivity, identify_samples_based_on_cq
from model_training_for_causal_pvr_task import load_dataset
import numpy as np
from utils.present_explanation import present_heatmap
import matplotlib.pylab as plt


def load_data_sample(dataset_path, dataset_name="mnist"):
    val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val")
    
    return val_dataset.__getitem__(random.randint(0,val_dataset.__len__()))

def show_input_sample(input_sample: torch.Tensor):
    transform = T.ToPILImage()
    img = transform(input_sample)
    img.show()

def generate_explanation(model:torch.nn.Module,input_img:torch.Tensor):
    last_layer_name = "layer4.1.bn2"
    input_numpy,cam, predict, class_idx = generate_grad_cam_from_img(last_layer_name, model, input_img)

    return present_heatmap(input_numpy, np.asarray(cam), predict, class_idx)

def generate_heatmap():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\\fork"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_sample, y, variables  = load_data_sample(dataset_path)

    print(f"{y} {variables}")

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()
    input_sample = input_sample.to(device)

    generate_explanation(model, input_sample)

def generate_cav():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
    dataset_name = "mnist"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"
    layer_name = "layer4.1.bn2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_dataset, val_dataset = load_dataset(
        dataset_path,
        dataset_name,
        batch_size=8, 
        is_return_dataset=True)

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()

    input_sample, y, variables  = load_data_sample(dataset_path)

    # Training CAV model
    train_cav_for_pvr_task(
        explained_model=model, 
        pvr_training_dataset=train_dataset,
        cav_save_path=os.path.join(dataset_path,"cavs"),
        layer_name = layer_name
    )

    # show_concept_dataset(
    #     os.path.join(dataset_path,"cavs"),
    #     layer_name = "layer4.1.bn2"
    # )

    # conceptual_sensitivity_dict, predict_index = calculate_local_cav_sensitivity(
    #     explained_model=model,
    #     explained_sample = input_sample,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    # sorted_conceptual_sensitivity_dict = sorted(conceptual_sensitivity_dict.items(), key=lambda kv:kv[1], reverse=True)

    # print(f"conceptual_sensitivity_dict: \n{sorted_conceptual_sensitivity_dict},\n predict_index: {predict_index},\n variables: {variables},\n")

    # identify_global_concept_causality_graph(
    #     explained_model=model, 
    #     pvr_training_dataset=train_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    # calculate_global_tcav(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    identify_samples_based_on_cav(
        explained_model=model,
        evaluate_dataset = val_dataset,
        cav_save_path=os.path.join(dataset_path,"cavs"),
        layer_name = layer_name,
        num_samples = 3,
        # explained_sample = input_sample,
    )

def generate_crp_relmax():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
    dataset_name = "mnist"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"
    layer_name = "layer4.1.conv1"
    target_channel = [50]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()

    _, _, train_dataset, val_dataset = load_dataset(
        dataset_path,
        dataset_name,
        batch_size=8, 
        is_return_dataset=True,
        return_pure_img=True)

    # input_sample, y, variables  = load_data_sample(dataset_path)
    input_sample = val_dataset.__getitem__(0)[0]

    input_sample = input_sample.to(device)

    # conditional_attributions(
    #     model, 
    #     input_sample, 
    #     layer_name
    #     )

    # feature_visualization(
    #     model, 
    #     val_dataset, 
    #     input_sample,
    #     layer_name)

    identify_graph(
        model, 
        input_sample,
        val_dataset,
        layer_name
    )

def generate_cexCNN():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
    dataset_name = "mnist"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"
    layer_name = "layer4.1.conv1"
    target_channel = [50]
    target_output = [1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()

    # _, _, train_dataset, val_dataset = load_dataset(
    #     dataset_path,
    #     dataset_name,
    #     batch_size=8, 
    #     is_return_dataset=True,
    #     return_pure_img=True)

    input_sample, y, variables  = load_data_sample(dataset_path)
    input_sample = input_sample.to(device)

    # filter_importance(model, input_sample, layer_name)

    weighted_map, important_map, importance = cexCNN_heatmap(model, input_sample, layer_name)

    present_heatmap(input_sample.cpu().numpy(), np.asarray(weighted_map))
    present_heatmap(input_sample.cpu().numpy(), np.asarray(important_map))

    print(importance)

def generate_concept_causal_map():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
    dataset_name = "mnist"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"
    layer_name = "layer4.1.conv1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, train_dataset, val_dataset = load_dataset(
        dataset_path,
        dataset_name,
        batch_size=1, 
        is_return_dataset=True)

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()

    input_sample, y, variables  = load_data_sample(dataset_path)

    i = 1
    while (i<4):
        input_sample, y, variables  = load_data_sample(dataset_path)
        if variables[0] == 0:
            plt.subplot(1,3,i)
            plt.imshow(input_sample.permute(1,2,0))
            plt.axis("off")

            i+=1
        
    plt.show()

    # plt.subplot(1,5,1)
    # plt.imshow(input_sample.permute(1,2,0))
    # plt.axis("off")
    # plt.subplot(1,5,2)
    # plt.imshow(input_sample.permute(1,2,0)[:28,:28,:])
    # plt.axis("off")
    # plt.subplot(1,5,3)
    # plt.imshow(input_sample.permute(1,2,0)[:28,28:,:])
    # plt.axis("off")
    # plt.subplot(1,5,4)
    # plt.imshow(input_sample.permute(1,2,0)[28:,:28,:])
    # plt.axis("off")
    # plt.subplot(1,5,5)
    # plt.imshow(input_sample.permute(1,2,0)[28:,28:,:])
    # plt.axis("off")
    # plt.show()

    # Training concept representation
    train_cp_for_pvr_task(
        explained_model=model,
        pvr_training_dataloader = train_dataloader,
        cp_save_path=os.path.join(dataset_path,"cp_position"),
        target_layer_type = [torch.nn.Conv2d],
        sample_num=200,
        pooling_type='mean'
    )

    # train_cp_for_pvr_task(
    #     explained_model=model,
    #     pvr_training_dataloader = train_dataloader,
    #     cp_save_path=os.path.join(dataset_path,"cp_position"),
    #     target_layer_type = [torch.nn.Conv2d],
    #     sample_num=200,
    #     pooling_type='mean'
    # )

    # show_concept_dataset_for_cp(
    #     os.path.join(dataset_path,"cp"),
    # )

    # contained_concept = concept_detection(
    #     explained_model = model,
    #     explained_sample = input_sample,
    #     cp_save_path = os.path.join(dataset_path,"cp_position"),
    #     pooling_type = 'mean'
    # )

    # conceptual_sensitivity_dict, predict_index = calculate_local_concept_sensitivity(
    #     explained_model = model,
    #     explained_sample = input_sample,
    #     cp_save_path = os.path.join(dataset_path,"cp_position"),
    #     pooling_type = 'mean', 
    #     contained_concept = contained_concept
    # )

    # print(conceptual_sensitivity_dict, predict_index, variables)


    # identify_global_concept_causality_graph(
    #     explained_model=model,
    #     cp_save_path=os.path.join(dataset_path,"cp_position"),
    #     num_samples = 200
    # )

    # calculate_global_concept_sensitivity(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    # identify_samples_based_on_cq(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name,
    #     num_samples = 5
    # )

def draw_figure_for_three_methods(sample_index:int=None):
    # This function is used to draw a figure containing three types of explanations (attribution methods, sample based methods, visualization methods) based on one input sample.

    # set up for model, dataset, input sample
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
    dataset_name = "mnist"
    model_parameter_path = os.path.join(dataset_path, "resnet_best.pt")
    num_class = 10
    model_name = "resnet"
    layer_name = "layer4.1.bn2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_dataset, val_dataset = load_dataset(
        dataset_path,
        dataset_name,
        batch_size=8, 
        is_return_dataset=True)

    model = load_model(model_name, model_parameter_path, num_class=num_class).to(device)
    model.eval()

    if sample_index == None:
        input_sample, y, variables = load_data_sample(dataset_path)
    else:
        input_sample, y, variables = val_dataset.__getitem__(sample_index)

    input_sample = input_sample.to(device)

    # # attribution methos
    # print("---attribution methos--------------------------------")
    # print(sample_index)
    # generate_explanation(model, input_sample)

    # # sample-based methos
    # print("---sample-based methos-------------------------------")
    # fig = identify_samples_based_on_cav(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name,
    #     num_samples = 8,
    #     explained_sample = input_sample,
    # )

    # fig.savefig(f"E:\\我的云端硬盘\\Phd_work\\conference_paper\\KDD2024\\Sample_{sample_index}.png")

    # visualization methos
    # print("---visualization methos------------------------------")
    # layer_name = "layer4.1.conv1"
    # identify_graph(
    #     model, 
    #     input_sample,
    #     val_dataset,
    #     layer_name
    # )

    # CoCa methods
    print("---CoCa methos------------------------------")
    contained_concept = concept_detection(
        explained_model = model,
        explained_sample = input_sample,
        cp_save_path = os.path.join(dataset_path,"cp_position"),
        pooling_type = 'mean'
    )

    conceptual_sensitivity_dict, predict_index = calculate_local_concept_sensitivity(
        explained_model = model,
        explained_sample = input_sample,
        cp_save_path = os.path.join(dataset_path,"cp_position"),
        pooling_type = 'mean', 
        contained_concept = contained_concept
    )

    print(conceptual_sensitivity_dict, predict_index, variables)



def sample_selection_methods():
    # CAV-based Methods
    fig = identify_samples_based_on_cav(
        explained_model=model,
        evaluate_dataset = val_dataset,
        cav_save_path=os.path.join(dataset_path,"cavs"),
        layer_name = layer_name,
        num_samples = 8,
        explained_sample = input_sample,
    )

    # Feature Similarity


    # 


if __name__ == "__main__":



    for i in range(100):
        generate_heatmap()
    
    # generate_cav()

    # generate_crp_relmax()

    # generate_cexCNN()

    # generate_concept_causal_map()



    # sample_indexs = [2,18,40,71]

    # for i in sample_indexs:
    #     draw_figure_for_three_methods(i)