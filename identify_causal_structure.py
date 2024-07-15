from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC, ANMNonlinear, DirectLiNGAM, ICALiNGAM, GES, Notears, TTPM, GAE, RL, GraNDAG

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from model_training_for_causal_pvr_task import load_dataset
from torch.utils.data import DataLoader

from causal_discovey.DiffAN.diffan import DiffAN


def load_data_for_consual_structure_identification(dataset_path, sample_num, is_random=False):
    '''
    concept_datasets:
        {
            "concept_name_1": {
                'input_sample_img':[], 
                'concept_label':[], 
                'concept_list':[], 
                'concept_input_features': {
                    'layer_1':[],
                    'layer_2':[],
                    ....
                    }
            },
            "concept_name_2": {
                'input_sample_img', 
                'concept_label', 
                'concept_list', 
                'concept_input_features'
            },
            ......
        }
    '''

    train_dataloader, val_dataloader, train_dataset, val_dataset = load_dataset(
        dataset_path,
        "mnist",
        batch_size=1, 
        is_return_dataset=True)

    if is_random:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True        
            )

    sample_dict = {}
    causal_sample_metrix = []

    for input_sample, model_label, concept_list  in val_dataloader:
        # print(concept_list.numpy().astype(np.float32))
        if model_label.item() in sample_dict:
            if len(sample_dict[model_label.item()]) < sample_num:
                sample_dict[model_label.item()].append(
                    {
                        "concept_labels": concept_list.squeeze().numpy()
                    }) 
                causal_sample_metrix.append(concept_list.squeeze().numpy().astype(np.float64))

        else:
            sample_dict[model_label.item()] = [
                {
                    "concept_labels": concept_list.squeeze().numpy()
                }
            ]
            causal_sample_metrix.append(concept_list.squeeze().numpy().astype(np.float64))

    causal_sample_metrix = np.matrix(causal_sample_metrix)
    
    return causal_sample_metrix
    

if __name__ == "__main__":

    dataset_path="F:\pvr_dataset\causal_validation_pvr_200sample\chain"

    causal_sample_metrix = load_data_for_consual_structure_identification(dataset_path, 100, False)

    # print("Start PC----------------------")
    # pc = PC()
    # pc.learn(causal_sample_metrix)
    # print(pc.causal_matrix)

    # print("Start ICALiNGAM----------------------")
    # icali = ICALiNGAM()
    # icali.learn(causal_sample_metrix)
    # print(icali.causal_matrix)

    # print("Start GAE----------------------")
    # gae = GAE()
    # gae.learn(causal_sample_metrix)
    # print(gae.causal_matrix)

    # print("Start RL----------------------")
    # rl = RL()
    # rl.learn(causal_sample_metrix)
    # print(rl.causal_matrix)

    # print("Start GraNDAG----------------------")
    # gradnag = GraNDAG()
    # gradnag.learn(causal_sample_metrix)
    # print(gradnag.causal_matrix)

    # plot predict_dag and true_dag
    # GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

    # calculate metrics
    # mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
    # print(mt.metrics)


    print("Start DiffAN----------------------")
    diffan = DiffAN(5,residue = False)
    adj_matrix, order = diffan.fit(causal_sample_metrix)
    # print(adj_matrix, order)
    print("----adj_matrix-------\n",adj_matrix)
    print("----order-------\n",order)
    

    
