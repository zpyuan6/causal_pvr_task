from data.CausalPVRDataset import CausalPVRDataset
import random
from model.model_training import load_model
import torchvision.transforms as T
import torch
import os
from explanation.heatmapes.grad_cam import generate_grad_cam_from_img
from explanation.concept_based.cav import train_cav_for_pvr_task, show_concept_dataset, calculate_local_cav_sensitivity, calculate_global_tcav, identify_samples_based_on_cav
from explanation.concept_based.crp_relmax import conditional_attributions, feature_visualization
from explanation.causal.cexCNN import filter_importance, cexCNN_heatmap
from model_training_for_causal_pvr_task import load_dataset
import numpy as np
from utils.present_explanation import present_heatmap


def load_data_sample(dataset_path, dataset_name="mnist"):
    val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val")
    
    return val_dataset.__getitem__(random.randint(0,val_dataset.__len__()))

def show_input_sample(input_sample: torch.Tensor):
    transform = T.ToPILImage()
    img = transform(input_sample)
    img.show()

def generate_explanation(model:torch.nn.Module,input_img:torch.Tensor):
    last_layer_name = "layer4.1.bn2"
    input_numpy,cam = generate_grad_cam_from_img(last_layer_name, model, input_img)

    present_heatmap(input_numpy, np.asarray(cam))

def generate_heatmap():
    dataset_path = "F:\pvr_dataset\causal_validation_pvr\chain"
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
    # train_cav_for_pvr_task(
    #     explained_model=model, 
    #     pvr_training_dataset=train_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    # show_concept_dataset(
    #     os.path.join(dataset_path,"cavs"),
    #     layer_name = "layer4.1.bn2"
    # )

    conceptual_sensitivity_dict, predict_index = calculate_local_cav_sensitivity(
        explained_model=model,
        explained_sample = input_sample,
        cav_save_path=os.path.join(dataset_path,"cavs"),
        layer_name = layer_name
    )

    print(conceptual_sensitivity_dict, predict_index, variables)

    # calculate_global_tcav(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name
    # )

    # identify_samples_based_on_cav(
    #     explained_model=model,
    #     evaluate_dataset = val_dataset,
    #     cav_save_path=os.path.join(dataset_path,"cavs"),
    #     layer_name = layer_name,
    #     num_samples = 5
    # )

def generate_crp_relmax():
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

    _, _, train_dataset, val_dataset = load_dataset(
        dataset_path,
        dataset_name,
        batch_size=8, 
        is_return_dataset=True,
        return_pure_img=True)

    input_sample, y, variables  = load_data_sample(dataset_path)

    input_sample = input_sample.unsqueeze(0).to(device)
    input_sample.requires_grad = True

    concept_ids = conditional_attributions(
        model, 
        input_sample, 
        layer_name, 
        target_channel, 
        target_output
        )

    feature_visualization(
        model, 
        val_dataset, 
        concept_ids,
        layer_name)

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


if __name__ == "__main__":
    # generate_heatmap()
    
    # generate_cav()

    # generate_crp_relmax()

    generate_cexCNN()