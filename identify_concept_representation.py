from model.model_training import load_model
import torch
import os
from model_training_for_causal_pvr_task import load_dataset
from explanation.concept_based.cav import train_cav_for_pvr_task_based_type, CAV, identify_samples_based_on_cav
from explanation.causal.concept_causal_map import train_cp_for_pvr_task, ConceptRepresentation, train_cp_for_pvr_task_for_causality, concept_maps, identify_samples_based_on_cp
import pandas as pd
import seaborn as sns
from multiple_explanation_examples import load_data_sample
from explanation.heatmapes.grad_cam import generate_grad_cam_from_img
import matplotlib.pyplot as plt

def train_cav_and_cr_for_causality(
    device,
    models:list,
    model_and_dataset_folder:str,
    dataset_types: list,
    dataset_name:str
):
    for dataset_type in dataset_types:
        for model_name in models:
            causal_type = dataset_type.split("_")[-1]

            print("Start {} model training".format(model_name))

            model = load_model(model_name, model_parameter_path=os.path.join(model_and_dataset_folder, dataset_type, f"{model_name}_best.pt"), num_class=10, is_print=True)
                
            model.to(device)

            train_dataloader, val_dataloader, train_dataset, val_dataset = load_dataset(
                os.path.join(model_and_dataset_folder,dataset_type.split("_")[-1]),
                dataset_name,
                batch_size=8, 
                is_return_dataset=True)

            train_cav_for_pvr_task_based_type(
                explained_model=model, 
                pvr_training_dataset=train_dataset,
                cav_save_path=os.path.join(model_and_dataset_folder,dataset_type, model_name, "cavs"),
                target_layer_type = [torch.nn.Conv2d],
                causal_type=causal_type
            )

            train_cp_for_pvr_task_for_causality(
                explained_model=model,
                pvr_training_dataloader = train_dataloader,
                cp_save_path=os.path.join(model_and_dataset_folder,dataset_type,model_name,"cp"),
                target_layer_type = [torch.nn.Conv2d],
                sample_num=10,
                pooling_type='mean',
                causal_type=causal_type
            )


def train_cav_and_cr(
    device,
    models:list,
    model_and_dataset_folder:str,
    dataset_types: list,
    dataset_name:str
):
    for dataset_type in dataset_types:
        for model_name in models:
            print("Start {} model training".format(model_name))

            model = load_model(model_name, model_parameter_path=os.path.join(model_and_dataset_folder, dataset_type, f"{model_name}_best.pt"), num_class=10, is_print=True)
                
            model.to(device)

            train_dataloader, val_dataloader, train_dataset, val_dataset = load_dataset(
                os.path.join(model_and_dataset_folder,dataset_type),
                dataset_name,
                batch_size=8, 
                is_return_dataset=True)

            train_cav_for_pvr_task_based_type(
                explained_model=model, 
                pvr_training_dataset=train_dataset,
                cav_save_path=os.path.join(model_and_dataset_folder,dataset_type, model_name, "cavs"),
                target_layer_type = [torch.nn.Conv2d]
            )

            train_cp_for_pvr_task(
                explained_model=model,
                pvr_training_dataloader = train_dataloader,
                cp_save_path=os.path.join(model_and_dataset_folder,dataset_type,model_name,"cp"),
                target_layer_type = [torch.nn.Conv2d],
                sample_num=10,
                pooling_type='mean'
            )





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ['vgg', 'resnet', 'densenet', 'mobilenet']
    model_and_dataset_folder = "F:\\pvr_dataset\\causal_validation_pvr_200sample"
    dataset_types = ['chain', 'collider', 'fork', 'limited_chain', 'limited_collider', 'limited_fork']
    # dataset_types = ['chain', 'collider', 'fork']
    dataset_name = "mnist"

    # train_cav_and_cr(
    #     device,
    #     models,
    #     model_and_dataset_folder,
    #     dataset_types,
    #     dataset_name
    # )

    train_cav_and_cr_for_causality(
        device,
        models,
        model_and_dataset_folder,
        dataset_types,
        dataset_name
    )