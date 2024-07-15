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

def generate_acc_table(data:pd.DataFrame):
    new_data = data[data['concept_value'] == 'overall']
    new_data = new_data[new_data['concept_type'] == 'cr']
    groups_data = new_data.groupby(['model','dataset_type','concept_type','concept']).agg({'acc':'max'})
    print(groups_data)

    groups_data = groups_data.groupby(['model','dataset_type','concept_type']).agg({'acc':'mean'})

    print(groups_data)


def demonstate_acc(data:pd.DataFrame):

    # new_data = data.drop(data.index[(data['concept_value'] == 'overall')])

    # sns.catplot(
    #     data=new_data, x="model", y="acc", hue="concept_type",
    #     kind="violin", split=True,
    # )
    # plt.show()
    
    # sns.catplot(data=new_data, x="model", y="acc", hue="concept_type", kind="bar")
    new_data = data[data['concept_value'] == 'overall']
    new_data = new_data[new_data['model'] == 'vgg']
    # new_data = new_data[new_data['concept'] == 'cc']
    # new_data = new_data[new_data['dataset_type'] == 'fork']
    new_data = new_data[new_data['dataset_type'] == 'limited_fork']
    for index, row in new_data.iterrows():
        new_data.loc[index,'layer'] = int(row['layer'].split(".")[-1])
        print(row['layer'])

    new_data = new_data.sort_values(by='layer')

    # groups_data = new_data.groupby(['model','dataset_type','concept_type','concept']).agg('max')
    # print(groups_data)

    # group_data = new_data.groupby(['concept'])
    # print(group_data)
    print(new_data)
    sns.catplot(data=new_data, x="layer", y="acc", hue="concept_type", kind="bar")

    # sns.lineplot(
    #     data=new_data, x="layer", y="acc", hue="concept_type", err_style="bars", errorbar=("se", 2),
    # )


    # sns.catplot(
    #     data=new_data, x="layer", y="acc", hue="concept_type",
    #     kind="violin", split=True,
    # )
    plt.show()

def storage_accuracy(
    models:list,
    model_and_dataset_folder:str,
    dataset_types: list,
):

    acc_dict = {}

    acc_dict['model'] = []
    acc_dict['layer'] = []
    acc_dict['dataset_type'] = []
    acc_dict['concept_type'] = []
    acc_dict['concept'] = []
    acc_dict['concept_value'] = []
    acc_dict['acc'] = []

    for dataset_type in dataset_types:
        for model_name in models:
            cav_save_path=os.path.join(model_and_dataset_folder,dataset_type, model_name, "cavs")

            for root, folders, files in os.walk(cav_save_path):
                for file in files:
                    if 'cav' in file:
                        cav = CAV()
                        cav.load_from_txt(os.path.join(root, file))

                        for concept_value in cav.accuracies.keys():
                            acc_dict['model'].append(model_name)
                            acc_dict['layer'].append(cav.bottleneck)
                            acc_dict['dataset_type'].append(dataset_type)
                            acc_dict['concept_type'].append('cav')
                            acc_dict['concept'].append(cav.concept_name)
                            acc_dict['concept_value'].append(concept_value)
                            acc_dict['acc'].append(cav.accuracies[concept_value])

                        print(cav.concept_name, cav.accuracies)


            cp_save_path=os.path.join(model_and_dataset_folder,dataset_type,model_name,"cp")

            for root, folders, files in os.walk(cp_save_path):
                for file in files:
                    if ('cp' in file) and ('best' not in file):
                        cr = ConceptRepresentation()
                        cr.load_from_txt(os.path.join(root, file))

                        for concept_value in cr.accuracies.keys():
                            acc_dict['model'].append(model_name)
                            acc_dict['layer'].append(cr.bottleneck)
                            acc_dict['dataset_type'].append(dataset_type)
                            acc_dict['concept_type'].append('cr')
                            acc_dict['concept'].append(cr.concept_name)
                            acc_dict['concept_value'].append(concept_value)
                            acc_dict['acc'].append(cr.accuracies[concept_value])

                        print(cr.concept_name, cr.accuracies)

    data = pd.DataFrame(acc_dict)
    data.to_csv(os.path.join(model_and_dataset_folder, "acc_statistic.csv"))

    return data
    

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

def generate_relevence_sample(
    device,
    models:list,
    model_and_dataset_folder:str,
    dataset_types: list,
    dataset_name:str
    ):

    dataset_type = dataset_types[0]
    model_name = models[0]

    model = load_model(model_name, model_parameter_path=os.path.join(model_and_dataset_folder, dataset_type, f"{model_name}_best.pt"), num_class=10, is_print=True)

    dataset_path = os.path.join(model_and_dataset_folder,dataset_type)


    train_dataloader, val_dataloader, train_dataset, val_dataset = load_dataset(
                dataset_path,
                dataset_name,
                batch_size=8, 
                is_return_dataset=True)

    cr_path = os.path.join(dataset_path,model_name,"cp")
    cav_path = os.path.join(dataset_path,model_name,"cavs")

    concept_list = ['a','b','c']

    value = {
        'a':[0,2],
        'b':[8],
        'c':[0,6],
        # 'd':[6,9]
    }
    # concept_list = ['cc']

    for root, folder, files  in os.walk(cr_path):
        for file in files:
            if 'best_cp' in file:
                cp = ConceptRepresentation()
                cp.load_from_txt(os.path.join(root, file))

                layer_name = cp.bottleneck

                for c in concept_list:
                    for v in value[c]:

                        identify_samples_based_on_cav(
                            explained_model=model,
                            evaluate_dataset = val_dataset,
                            cav_save_path=cav_path,
                            layer_name = layer_name,
                            concept_name = c,
                            concept_target_value = v,
                            num_samples = 5,
                        )

                        # identify_samples_based_on_cp(
                        #     explained_model=model,
                        #     evaluate_dataset = val_dataset,
                        #     cp_save_path=cr_path,
                        #     layer_name = layer_name,
                        #     concept_name = c,
                        #     concept_target_value = v,
                        #     num_samples = 4
                        # )
    

                


        
    



def draw_conceptmap(
    models,
    model_and_dataset_folder,
    dataset_types,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # resnet
    model_name = models[1]

    
    for dataset_type in dataset_types:
        input_sample, y, variables  = load_data_sample(os.path.join(model_and_dataset_folder, dataset_type))

        model = load_model(model_name, model_parameter_path=os.path.join(model_and_dataset_folder, dataset_type, f"{model_name}_best.pt"), num_class=10, is_print=True)
                
        model.to(device)
        model.eval()
        input_sample = input_sample.to(device)

        last_layer_name = "layer4.1.bn2"
        input_numpy, cam, predict, class_idx = generate_grad_cam_from_img(last_layer_name, model, input_sample)

        concept_map_list, concept_detection_results = concept_maps(
            model,
            input_sample,
            os.path.join(model_and_dataset_folder, dataset_type,"cp_position"),
            pooling_type='mean'
        )

        cams = [cam] + concept_map_list
        titles = ["GradCam"] + concept_detection_results

    for i in range(len(cams)):
        if input_sample.shape[-1]!=3:
            input_sample = input_sample.permute(1,2,0).to('cpu')

        plt.subplot(1, len(cams), i+1)
        plt.imshow(input_sample)
        plt.imshow(cams[i], alpha=0.5)
        plt.title(titles[i])
        plt.axis('off')

    plt.show()

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

    # train_cav_and_cr_for_causality(
    #     device,
    #     models,
    #     model_and_dataset_folder,
    #     dataset_types,
    #     dataset_name
    # )

    # acc = storage_accuracy(
    #     models,
    #     model_and_dataset_folder,
    #     dataset_types,
    # )
    # acc = pd.read_csv(os.path.join(model_and_dataset_folder, "acc_statistic.csv"))

    # generate_acc_table(acc)

    generate_relevence_sample(
        device,
        models,
        model_and_dataset_folder,
        dataset_types,
        dataset_name
    )

    # demonstate_acc(acc)

    # for i in range(100):
    #     draw_conceptmap(
    #         models,
    #         model_and_dataset_folder,
    #         dataset_types
    #     )

    
