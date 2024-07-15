import torch
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset
from torch.nn import Module
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class CAV(object):
    """CAV class contains methods for concept activation vector (CAV).

    CAV represents semenatically meaningful vector directions in
    network's embeddings (bottlenecks).
    """
    def __init__(
        self, 
        concept_name=None, 
        bottleneck=None, 
        hparams=None, 
        save_path=None):
        """Initialize CAV class.

        Args:
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          hparams: a parameter used to learn CAV
            {
                model_type:''
                alpha:''
            }
          save_path: where to save this CAV
        """
        self.concept_name = concept_name
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path

    def train(self, x, label):
        """Train the CAVs from the activations.

        Args:
          x: 
          label:
        Raises:
          ValueError: if the model_type in hparam is not compatible.
        """

        print('training with alpha={}'.format(self.hparams["alpha"]))

        if self.hparams["model_type"] == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams["alpha"], tol=1e-3, max_iter=1000)
        elif self.hparams["model_type"] == 'logistic':
            lm = linear_model.LogisticRegression()
        else:
            raise ValueError('Invalid hparams.model_type: {}'.format(self.hparams["model_type"]))

        self.accuracies = self._train_lm(lm, x, label)
        if len(lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self._save_cavs()

    def get_direction(self, index):
        return self.cavs[index]


    def _train_lm(self, lm, x:torch.tensor, y:torch.tensor):
        """Train a model to get CAVs.

        Modifies lm by calling the lm.fit functions. The cav coefficients are then
        in lm._coefs.

        Args:
          lm: An sklearn linear_model object. Can be linear regression or
            logistic regression. Must support .fit and ._coef.
          x: An array of training data of shape [num_data, data_dim]
          y: An array of integer labels of shape [num_data]
          labels2text: Dictionary of text for each label.

        Returns:
          Dictionary of accuracies of the CAVs.

        """
        print(x.shape, y.shape, type(x), type(y))

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y)
        # if you get setting an array element with a sequence, chances are that your
        # each of your activation had different shape - make sure they are all from
        # the same layer, and input image size was the same
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        # get acc for each class.
        num_classes = int(max(y) + 1)
        acc = {}
        num_correct = 0
        for class_id in range(num_classes):
            # get indices of all test data that has this class.
            idx = (y_test == class_id)
            acc[class_id] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[class_id])
        acc['overall'] = float(num_correct) / float(len(y_test))
        print('acc per class %s' % (str(acc)))
        return acc

    def _save_cavs(self):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concept_name': self.concept_name,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            print('save_path is None. Not saving anything')

    def load_from_txt(self, save_path=None):
        if not save_path==None:
            self.save_path = save_path
        
        if not os.path.exists(self.save_path):
            raise Exception(f"Can not found file in path {self.save_path}")

        with open(self.save_path, 'rb') as file:
            cav = pickle.load(file)
            self.concept_name = cav["concept_name"]
            self.bottleneck = cav["bottleneck"]
            self.hparams = cav["hparams"]
            self.accuracies = cav["accuracies"]
            self.cavs = cav["cavs"]

        return self


features_out = []

def cav_register_hook(model:Module, layer_name:str):
    def forward_hook(module, input, output):
        features = output.reshape([output.shape[0],-1]).cpu().detach().numpy()
        features_out.append(features)
        return None

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            # print("find layer ", name)
            hooks.append(module.register_forward_hook(forward_hook))

    return hooks

grad_out = []

def register_gradient_hook(model:Module, layer_name:str):
    def backward_hook(module, grad_input, grad_output):
        # print("backward_hook")
        grad = grad_output[0].reshape([grad_output[0].shape[0],-1]).cpu().detach().numpy()
        grad_out.append(grad)
        return None

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            # print("find layer ", name)
            hooks.append(module.register_backward_hook(backward_hook))

    return hooks

def construct_pvr_concept_dataset_for_causality(
    pvr_dataset:Dataset, 
    sample_num:int, 
    model:Module, 
    layer_name:str,
    causal_type:str
    ):

    """ 
    
    Return:
        concept_dataset_dict: is a dict in format 
            {
                'concept name 1': 
                (
                    original model input numpy,
                    features numpy for target layer,
                    concept annotation value
                ),
                'concept name 1': (...),
                ...
            }
    """

    causality_types = ["chain", "fork", "collider"]

    if causal_type == causality_types[0]:
        concept_list = ['ca','cb']
    elif causal_type == causality_types[1]:
        concept_list = ['ca','cb','cc']
    elif causal_type == causality_types[2]:
        concept_list = ['ca','cb','cc']
    else:
        raise Exception(f"Can not find causality type {causal_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concept_dataset_dict = {}

    model.to(device)

    with tqdm(total= len(concept_list), desc="Sperate dataset to concept dataset") as tbar:
        for concept in concept_list:
            tbar.update(1)
            concept_index = concept_list.index(concept)

            data = {}
            for i in range(pvr_dataset.__len__()):
                input_x, output_y, all_value = pvr_dataset.__getitem__(i)
                all_value =  all_value.numpy()
                if concept_index == 0:
                    # ca sample
                    if all_value[0] in data:
                        if len(data[all_value[0]]) < sample_num:
                            data[all_value[0]].append(input_x)
                    else:
                        data[all_value[0]] = [input_x]
                elif concept_index == 1:
                    # cb sample
                    if all_value[0] < 4:
                        cb_value = 0 
                    elif all_value[0] < 7:
                        cb_value = 1
                    else:
                        cb_value = 2

                    if cb_value in data:
                        if len(data[cb_value]) < sample_num:
                            data[cb_value].append(input_x)
                    else:
                        data[cb_value] = [input_x]
                elif concept_index == 2:
                    # cc sample
                    if causal_type == causality_types[1]:
                        if all_value[0] < 4:
                            cc_value = all_value[2]
                        elif all_value[0] < 7:
                            cc_value = all_value[3]
                        else:
                            cc_value = all_value[1]
                        if cc_value in data:
                            if len(data[cc_value]) < sample_num:
                                data[cc_value].append(input_x)
                        else:
                            data[cc_value] = [input_x]
                    elif causal_type == causality_types[2]:
                        if all_value[0] < 4:
                            cc_value = 1
                        elif all_value[0] < 7:
                            cc_value = 2
                        else:
                            cc_value = 0
                        if cc_value in data:
                            if len(data[cc_value]) < sample_num:
                                data[cc_value].append(input_x)
                        else:
                            data[cc_value] = [input_x]
                    else:
                        raise Exception(f"Can not find causality type {causal_type}")
                else:
                    raise Exception(f"Concept_index out off range")

            hooks = cav_register_hook(model, layer_name)
            
            values = []
            input_samples = []

            for key, input_sample in data.items():
                values.extend([key for i in range(len(input_sample))])
                input_samples.extend(input_sample)
            
            input_tensor = torch.stack(input_samples)
            print(input_tensor.shape, layer_name)
            model_input = input_tensor.to(device)

            model(model_input)

            concept_dataset_dict[concept] = (input_tensor.numpy(), features_out[0], np.array(values).astype(np.float32))

            for h in hooks:
                h.remove()

            features_out.clear()

    return concept_dataset_dict


def construct_pvr_concept_dataset(
    pvr_dataset:Dataset, 
    concept_list:list, 
    sample_num:int, 
    model:Module, 
    layer_name:str
    ):

    """ 
    
    Return:
        concept_dataset_dict: is a dict in format 
            {
                'concept name 1': 
                (
                    original model input numpy,
                    features numpy for target layer,
                    concept annotation value
                ),
                'concept name 1': (...),
                ...
            }
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concept_dataset_dict = {}

    model.to(device)

    with tqdm(total= len(concept_list), desc="Sperate dataset to concept dataset") as tbar:
        for concept in concept_list:
            tbar.update(1)
            # position, value = concept.split("_")
            # value = int(value)
            position_index = concept_list.index(concept)

            # positive_training_samples = []
            # negative_training_samples = []

            data = {}
            for i in range(pvr_dataset.__len__()):
                input_x, output_y, all_value = pvr_dataset.__getitem__(i)
                all_value =  all_value.numpy()
                if all_value[position_index] in data:
                    if len(data[all_value[position_index]]) < sample_num:
                        data[all_value[position_index]].append(input_x)
                else:
                    data[all_value[position_index]] = [input_x]

            hooks = cav_register_hook(model, layer_name)
            
            values = []
            input_samples = []

            for key, input_sample in data.items():
                values.extend([key for i in range(len(input_sample))])
                input_samples.extend(input_sample)
            
            input_tensor = torch.stack(input_samples)
            print(input_tensor.shape, layer_name)
            model_input = input_tensor.to(device)

            model(model_input)

            concept_dataset_dict[concept] = (input_tensor.numpy(), features_out[0], np.array(values).astype(np.float32))

            for h in hooks:
                h.remove()

            features_out.clear()

    return concept_dataset_dict


def train_cav_for_pvr_task_based_type(
    explained_model:torch.nn.Module, 
    pvr_training_dataset:Dataset,
    cav_save_path:str,
    target_layer_type:list,
    causal_type:str = None):

    target_layer_names = []

    for name, layer in explained_model.named_modules():
        for layer_definition in target_layer_type:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in target_layer_names:
                    target_layer_names.append(name)
                    train_cav_for_pvr_task(
                                explained_model, 
                                pvr_training_dataset,
                                cav_save_path,
                                name,
                                causal_type)

def train_cav_for_pvr_task(
    explained_model:torch.nn.Module, 
    pvr_training_dataset:Dataset,
    cav_save_path:str,
    layer_name:str,
    causal_type:str = None):

    if not os.path.exists(cav_save_path):
        os.makedirs(cav_save_path)

    # concept_cav_list = []
    position_list = ['a','b','c','d']
    sample_num = 10

    # for position in position_list:
    #     for i in range(10):
    #         concept_cav_list.append(f"{position}_{i}")

    if causal_type== None:
        concept_datasets = construct_pvr_concept_dataset(
            pvr_training_dataset, 
            position_list, 
            sample_num,
            explained_model, 
            layer_name
            )
    else:
        concept_datasets = construct_pvr_concept_dataset_for_causality(
            pvr_training_dataset, 
            sample_num,
            explained_model, 
            layer_name,
            causal_type
            )

    pickle.dump(concept_datasets, open(os.path.join(cav_save_path,f"concept_dataset_causality_{layer_name}.txt"), 'wb'))

    for concept_name in concept_datasets.keys():
        print(concept_name)
        model_input, concept_input, concept_value = concept_datasets[concept_name]

        cav = CAV(
            concept_name, 
            layer_name, 
            {
                "model_type":'linear',
                "alpha":0.01
                }, 
            save_path=os.path.join(cav_save_path, f"cav_for_{concept_name}_{layer_name}.txt")
        )

        cav.train(concept_input,concept_value)

def show_concept_dataset(
    cav_save_path:str,
    layer_name:str
    ):

    concept_dataset = pickle.load(open(os.path.join(cav_save_path,f"concept_dataset_{layer_name}.txt"), 'rb'))

    for concept_name in concept_dataset.keys():
        model_input, features_input, concept_label = concept_dataset[concept_name]

        plt.subplot(1,2,1)
        plt.imshow(model_input[0].transpose(1,2,0))
        plt.title(concept_name)
        plt.subplot(1,2,2)
        plt.imshow(model_input[-1].transpose(1,2,0))
        plt.show()
        
def calculate_local_cav_sensitivity(
        explained_model: torch.nn.Module,
        explained_sample: torch.Tensor,
        cav_save_path: str,
        layer_name: str,
    ) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register hook
    backward_hooks = register_gradient_hook(explained_model, layer_name)

    # Forward process
    model_input = explained_sample.unsqueeze(0).to(device)
    explained_model = explained_model.to(device)
    output = explained_model(model_input)
    predict_index = torch.argmax(torch.softmax(output,1)[0])

    # Backward process
    one_hot = F.one_hot(predict_index,num_classes=output.size()[-1])
    one_hot = one_hot.float().requires_grad_(True)
    one_hot = torch.sum(one_hot * output) 
    explained_model.zero_grad()
    one_hot.backward(retain_graph=True)

    # Get gradient
    grad = grad_out[-1].squeeze()
    
    # Get conceptual sensitivity
    conceptual_sensitivity_dict = {}

    for root, folder, files in os.walk(cav_save_path):
        for file in files:
            if file.split("_")[0] == "cav":
                cav = CAV()
                cav.load_from_txt(os.path.join(root,file))

                if layer_name != cav.bottleneck:
                    raise Exception(f"You input a different layer name {layer_name} for target cav layer name {cav.bottleneck}")

                conceptual_sensitivity = - np.dot(grad, cav.get_direction())

                conceptual_sensitivity_dict[cav.concept_name] = conceptual_sensitivity

    for h in backward_hooks:
        h.remove()

    grad_out.clear()
                
    return conceptual_sensitivity_dict, predict_index


def calculate_global_tcav(
        explained_model: torch.nn.Module,
        evaluate_dataset: Dataset,
        cav_save_path: str,
        layer_name: str,
    ):

    global_tcav_dict = {}

    with tqdm(total=evaluate_dataset.__len__()) as tbar:
        tbar.set_description_str("Calculating local local cav")
        for i in range(evaluate_dataset.__len__()):
            tbar.update(1)

            explained_sample, classification, record = evaluate_dataset.__getitem__(i)

            local_cav_sensitivity, predict_index = calculate_local_cav_sensitivity(
                explained_model,
                explained_sample,
                cav_save_path,
                layer_name)

            predict_index = predict_index.cpu().item()

            if not (predict_index in global_tcav_dict):
                global_tcav_dict[predict_index] = {}

            for concept_name in local_cav_sensitivity.keys():
                if concept_name in global_tcav_dict[predict_index]:
                    global_tcav_dict[predict_index][concept_name].append(local_cav_sensitivity[concept_name])
                else:
                    global_tcav_dict[predict_index][concept_name] = [local_cav_sensitivity[concept_name]]

    for label in global_tcav_dict.keys():

        for concept in global_tcav_dict[label].keys():
            cav_sensitive_list = global_tcav_dict[label][concept]
            effective_sample = np.array(cav_sensitive_list)>0
            global_tcav_dict[label][concept] = np.sum(effective_sample) / len(cav_sensitive_list)
    
    print(global_tcav_dict)

    return global_tcav_dict

def identify_samples_based_on_cav(
    explained_model: torch.nn.Module,
    evaluate_dataset: Dataset,
    cav_save_path: str,
    layer_name: str,
    num_samples: int,
    concept_name: str = None,
    concept_target_value: int  = None,
    explained_sample: torch.Tensor = None,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explained_model = explained_model.to(device)

    target_concept_list = []

    if concept_name!= None:
        target_concept_list.append(concept_name)

    if explained_sample != None:
        conceptual_sensitivity_dict, predict_index = calculate_local_cav_sensitivity(
            explained_model=explained_model,
            explained_sample = explained_sample,
            cav_save_path=cav_save_path,
            layer_name = layer_name
        )

        sorted_conceptual_sensitivity_dict = sorted(conceptual_sensitivity_dict.items(), key=lambda kv:kv[1], reverse=True)[:4]

        print(f"conceptual_sensitivity_dict: \n{sorted_conceptual_sensitivity_dict},\n predict_index: {predict_index}")

        for item in sorted_conceptual_sensitivity_dict:
            target_concept_list.append(item[0])


    hooks = cav_register_hook(explained_model, layer_name)

    if len(features_out) == 0:
        with tqdm(total=evaluate_dataset.__len__()) as tbar:
            tbar.set_description_str("Calculating local local cav")
            for i in range(evaluate_dataset.__len__()):
                tbar.update(1)

                explained_sample, classification, record = evaluate_dataset.__getitem__(i)
                model_input = explained_sample.unsqueeze(0).to(device)
                explained_model = explained_model.to(device)
                output = explained_model(model_input)


    features = features_out

    similar_sample_dict={}
    
    for root, folder, files in os.walk(cav_save_path):
        for file in files:
            if file.split("_")[0] == "cav":
                cav = CAV()
                cav.load_from_txt(os.path.join(root,file))

                if len(target_concept_list)==0 or (cav.concept_name in target_concept_list):

                    if layer_name != cav.bottleneck:
                        continue

                    target_cav = cav

                    cosine_similarity_list = []

                    for feature in features:
                        cosine_similarity = np.dot(feature.squeeze(), cav.get_direction(concept_target_value)) / (np.linalg.norm(feature.squeeze())*np.linalg.norm(cav.get_direction(concept_target_value)))
                        cosine_similarity_list.append(cosine_similarity)

                    cosine_similarity_list = np.array(cosine_similarity_list)
                    top_index = cosine_similarity_list.argsort()[::-1]

                    similar_sample_dict[cav.concept_name] = top_index

    for i in range(num_samples):
        plt.subplot(1, num_samples*2, i+1)
        show_sample,_,_ = evaluate_dataset.__getitem__(top_index[i])
        show_sample = show_sample.numpy().transpose(1,2,0)
        plt.imshow(show_sample)
        plt.axis('off')
        # plt.title(f"Sim:{round(cosine_similarity_list[top_index[i]],2)}")

    for i in range(num_samples):
        plt.subplot(1, num_samples*2, i+1+num_samples)
        show_sample,_,_ = evaluate_dataset.__getitem__(top_index[-i-1])
        show_sample = show_sample.numpy().transpose(1,2,0)
        plt.imshow(show_sample)
        plt.axis('off')


    plt.suptitle(f"CAV: {target_cav.concept_name} {concept_target_value}\nAcc:{target_cav.accuracies['overall']}")

    plt.show()

    [h.remove() for h in hooks]
    # features_out.clear()

    # fig, axes = plt.subplots(nrows = len(list(similar_sample_dict.keys())), ncols = num_samples, figsize=(num_samples*2,len(list(similar_sample_dict.keys()))*2))

    # for concept in similar_sample_dict.keys():
    #     concept_index = list(similar_sample_dict.keys()).index(concept)
        
    #     for i in range(num_samples):
    #         # axes[concept_index,i].imshow()
    #         # plt.subplot(concept_index+1, num_samples, i+1)
    #         show_sample,_,_ = evaluate_dataset.__getitem__(top_index[i])
    #         show_sample = show_sample.numpy().transpose(1,2,0)
    #         axes[concept_index,i].imshow(show_sample)
    #         axes[concept_index,i].axis('off')
    #         axes[concept_index,i].set_title(f"Sim:{round(cosine_similarity_list[top_index[i]].item(),4)}")

    #     print(concept, cav.accuracies)
        
    #     axes[concept_index,0].axis('on')
    #     axes[concept_index,0].set_yticks([])
    #     axes[concept_index,0].set_xticks([])
    #     axes[concept_index,0].set_ylabel(f"Concept: {concept}", rotation=0, size='large',
    #                ha='right', va='center')
    #     # plt.suptitle(f"{cav.concept_name}\nAcc:{cav.accuracies['overall']}")

    # plt.show()

    # for h in hooks:
    #     h.remove()

    # features_out.clear()

    # return fig





                


