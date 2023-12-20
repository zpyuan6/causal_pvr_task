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

    def get_direction(self):
        return self.cavs[0]


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


def construct_pvr_concept_dataset(
    pvr_dataset:Dataset, 
    concept_list:list, 
    position_list:list, 
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

    for concept in concept_list:
        position, value = concept.split("_")
        value = int(value)
        position_index = position_list.index(position)

        positive_training_samples = []
        negative_training_samples = []

        for i in range(pvr_dataset.__len__()):
            input_x, output_y, all_value = pvr_dataset.__getitem__(i)
            if all_value[position_index] == value:
                if len(positive_training_samples) < sample_num:
                    positive_training_samples.append((input_x,1))
            else:
                if len(negative_training_samples) < sample_num:
                    negative_training_samples.append((input_x,0))

            if len(positive_training_samples) == sample_num and len(negative_training_samples) == sample_num:
                break 

        training_samples = positive_training_samples + negative_training_samples

        cav_register_hook(model, layer_name)
        
        values = []
        input_samples = []
        for item in training_samples:
            input_sample, concept_value = item
            values.append(concept_value)
            input_samples.append(input_sample)
        
        input_tensor = torch.stack(input_samples)

        model_input = input_tensor.to(device)

        hooks = cav_register_hook(model, layer_name)

        model(model_input)

        concept_dataset_dict[concept] = (input_tensor.numpy(), features_out[0], np.array(values).astype(np.float32))

        for h in hooks:
            h.remove()

        features_out.clear()

    return concept_dataset_dict


def train_cav_for_pvr_task(
    explained_model:torch.nn.Module, 
    pvr_training_dataset:Dataset,
    cav_save_path:str,
    layer_name:str):

    if not os.path.exists(cav_save_path):
        os.makedirs(cav_save_path)

    concept_cav_list = []
    position_list = ['a','b','c','d']
    sample_num = 200

    for position in position_list:
        for i in range(10):
            concept_cav_list.append(f"{position}_{i}")

    concept_datasets = construct_pvr_concept_dataset(
        pvr_training_dataset, 
        concept_cav_list, 
        position_list, 
        sample_num,
        explained_model, 
        layer_name
        )

    pickle.dump(concept_datasets, open(os.path.join(cav_save_path,f"concept_dataset_{layer_name}.txt"), 'wb'))

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
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explained_model = explained_model.to(device)

    cav_register_hook(explained_model, layer_name)

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

                if layer_name != cav.bottleneck:
                    raise Exception(f"You input a different layer name {layer_name} for target cav layer name {cav.bottleneck}")

                cosine_similarity_list = []

                for feature in features:
                    cosine_similarity = - np.dot(feature.squeeze(), cav.get_direction()) / (np.linalg.norm(feature.squeeze())*np.linalg.norm(cav.get_direction()))
                    cosine_similarity_list.append(cosine_similarity)

                cosine_similarity_list = np.array(cosine_similarity_list)
                top_index = cosine_similarity_list.argsort()[::-1]

                similar_sample_dict[cav.concept_name] = top_index

                for i in range(num_samples):
                    plt.subplot(1, num_samples, i+1)
                    show_sample,_,_ = evaluate_dataset.__getitem__(top_index[i])
                    show_sample = show_sample.numpy().transpose(1,2,0)
                    plt.imshow(show_sample)
                    plt.title(f"Sim:{round(cosine_similarity_list[top_index[i]],3)}")

                    print(cav.accuracies)

                plt.suptitle(f"{cav.concept_name}\nAcc:{cav.accuracies['overall']}")

                plt.show()





                


