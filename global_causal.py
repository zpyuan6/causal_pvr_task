from data.CausalPVRDataset import CausalPVRDataset
import random
from model.model_training import load_model
import torchvision.transforms as T
import torch
import os
from explanation.causal.concept_causal_map import train_cp_for_pvr_task, show_concept_dataset_for_cp, calculate_local_concept_sensitivity, concept_detection, calculate_local_concept_sensitivity_based_on_gradient, identify_global_concept_causality_graph, identify_local_concept_causality_graph
# calculate_global_concept_sensitivity, identify_samples_based_on_cq
from model_training_for_causal_pvr_task import load_dataset
import numpy as np
import matplotlib.pylab as plt

def load_data_sample(dataset_path, dataset_name="mnist"):
    val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val")
    
    return val_dataset.__getitem__(random.randint(0,val_dataset.__len__()))

def generate_concept_causal_map(
    dataset_path:str,
    model_name:str="mnist"
    ):

    causal_type = dataset_path.split('\\')[-1].split("_")[-1]

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

    print(f"Input Sample {variables}")
    # i = 1
    # while (i<4):
    #     input_sample, y, variables  = load_data_sample(dataset_path)
    #     if variables[0] == 0:
    #         plt.subplot(1,3,i)
    #         plt.imshow(input_sample.permute(1,2,0))
    #         plt.axis("off")

    #         i+=1

    # plt.show()

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
    #     cp_save_path=os.path.join(dataset_path,model_name,"cp"),
    #     num_samples = 10,
    #     causal_type =causal_type
    # )

    identify_global_concept_causality_graph(
        explained_model=model,
        cp_save_path=os.path.join(dataset_path,"cp_position"),
        num_samples = 200,
        causal_type =causal_type
    )

    # identify_local_concept_causality_graph(
    #     explained_model=model,
    #     explained_sample = input_sample,
    #     cp_save_path=os.path.join(dataset_path,"cp_position"),
    #     pooling_type = 'mean', 
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


if __name__ == "__main__":

    generate_concept_causal_map(dataset_path="F:\pvr_dataset\causal_validation_pvr_200sample\chain")

    # generate_concept_causal_map(dataset_path="F:\pvr_dataset\causal_validation_pvr\chain")

    # generate_concept_causal_map(dataset_path="F:\pvr_dataset\causal_validation_pvr\limited_chain")

    # generate_concept_causal_map(dataset_path="F:\\pvr_dataset\\causal_validation_pvr\\fork")

    # generate_concept_causal_map(dataset_path="F:\\pvr_dataset\\causal_validation_pvr\\limited_fork")

    # generate_concept_causal_map(dataset_path="F:\\pvr_dataset\\causal_validation_pvr\\collider")

    # generate_concept_causal_map(dataset_path="F:\pvr_dataset\causal_validation_pvr\limited_collider")