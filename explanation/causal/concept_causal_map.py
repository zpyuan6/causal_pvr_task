import torch
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import pandas as pd
from pyvis.network import Network

class ConceptRepresentation(object):
    """CAV class contains methods for concept activation vector (CAV).

    CAV represents semenatically meaningful vector directions in
    network's embeddings (bottlenecks).
    """
    def __init__(
        self, 
        concept_name:str=None, 
        concept_list:list=None,
        bottleneck:str=None, 
        hparams:dict=None, 
        save_path:str=None):
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
        self.concept_list = concept_list
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
            self.cps = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cps = [c for c in lm.coef_]

        self.lm = lm

        self.save_cp()

    def get_direction(self, concept):
        return self.cps[self.concept_list.index(concept)]

    def predict(self, input_feature):
        y_pred = self.lm.predict(input_feature)
        return y_pred

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
        print(self.concept_name, self.concept_list, x.shape, y.shape)

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
        print(f'acc per class {str(acc)} {self.bottleneck}')
        return acc

    def save_cp(self):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concept_name': self.concept_name,
            'concept_list': self.concept_list,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cps': self.cps,
            'save_path': self.save_path,
            'lm': self.lm
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
            cp = pickle.load(file)
            self.concept_name = cp["concept_name"]
            self.bottleneck = cp["bottleneck"]
            self.concept_list = cp["concept_list"]
            self.hparams = cp["hparams"]
            self.accuracies = cp["accuracies"]
            self.cps = cp["cps"]
            self.save_path = cp['save_path']
            self.lm = cp['lm']

        return self


features_out = []

def cp_register_hook(model:Module, layer_names:list, pooling_type:str):
    if not pooling_type in ['mean','max']:
        raise Exception(f"Can not support pooling type {pooling_type}")

    def forward_hook(module, input, output):
        if pooling_type == 'mean':
            features = torch.mean(output.reshape([output.shape[0],output.shape[1],-1]), dim=-1).cpu().detach().numpy()
            # features = torch.nn.ReLU()(torch.mean(output.reshape([output.shape[0],output.shape[1],-1]), dim=-1)).cpu().detach().numpy()
        elif pooling_type == 'max':
            features, max_indices = torch.max(output.reshape([output.shape[0],output.shape[1],-1]), dim=-1)
            features = features.cpu().detach().numpy()

        features_out.append(features)
        return None

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name in layer_names:
            # print("find layer ", name)
            hooks.append(module.register_forward_hook(forward_hook))

    return hooks



def register_hook_for_do_calculate(model:Module, layer_name:str, do_value, pooling_type:str):

    def forward_hook(module, input, output):

        if pooling_type=="mean":
            new_do_value = do_value.unsqueeze(-1).unsqueeze(-1)
        elif pooling_type == "max":
            new_do_value = do_value.unsqueeze(-1).unsqueeze(-1) + torch.zeros(output.shape).to(output.device)
            max_value, _ = output.reshape([output.shape[1],-1]).max(dim=-1, keepdim=True)
            calculate_output = output.reshape([output.shape[1],-1])
            mask = (max_value==calculate_output)
            mask = mask.reshape(output.shape)
            new_do_value = new_do_value * mask
        new_output = output - new_do_value

        return new_output

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            hooks.append(module.register_forward_hook(forward_hook))

    return hooks


grad_out = []

def register_gradient_hook_for_cq(model:Module, layer_names:list):
    def backward_hook(module, grad_input, grad_output):
        grad = grad_output
        print(grad[0].shape)
        grad_out.append(grad[0])
        return None

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name in layer_names:
            # print("find layer ", name)
            hooks.append((name, module.register_backward_hook(backward_hook)))

    return hooks

def construct_pvr_concept_dataset_for_cp(
    pvr_datasetloader:DataLoader, 
    concept_list:list, 
    position_list:list, 
    sample_num:int, 
    model:Module, 
    target_layer_names:list,
    pooling_type:str
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
            position, value = concept.split("_")
            value = int(value)
            position_index = position_list.index(position)

            positive_training_samples = []
            negative_training_samples = []

            for epoch in pvr_datasetloader:
                input_x, output_y, all_value = epoch
                input_x, output_y, all_value = input_x[0], output_y[0], all_value[0]
                if all_value[position_index] == value:
                    if len(positive_training_samples) < sample_num:
                        positive_training_samples.append((input_x,1))
                else:
                    if len(negative_training_samples) < sample_num:
                        negative_training_samples.append((input_x,0))

                if len(positive_training_samples) == sample_num and len(negative_training_samples) == sample_num:
                    break 

            training_samples = positive_training_samples + negative_training_samples

            hooks = cp_register_hook(model, target_layer_names, pooling_type)

            values = []
            input_samples = []
            for item in training_samples:
                input_sample, concept_value = item
                values.append(concept_value)
                input_samples.append(input_sample)
            
            input_tensor = torch.stack(input_samples)
            model_input = input_tensor.to(device)
            model(model_input)

            concept_dataset_dict[concept] = {"input_sample_img":input_tensor.numpy(), "concept_label":np.array(values).astype(np.float32), "concept_input_features": {}}

            for i, layer_name in enumerate(target_layer_names):
                concept_dataset_dict[concept]["concept_input_features"][layer_name] = features_out[i]

            for h in hooks:
                h.remove()

            features_out.clear()

    return concept_dataset_dict


def construct_pvr_position_dataset(
    pvr_datasetloader:DataLoader, 
    position_list:list, 
    sample_num:int, 
    model:Module, 
    target_layer_names:list,
    pooling_type:str
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

    with tqdm(total= len(position_list), desc="Sperate dataset to concept dataset") as tbar:
        for position in position_list:
            tbar.update(1)
            position_index = position_list.index(position)

            data = {}
            for epoch in pvr_datasetloader:
                input_x, output_y, all_value = epoch
                input_x, output_y, all_value = input_x[0], output_y[0], all_value[0].numpy()
                if all_value[position_index] in data:
                    if len(data[all_value[position_index]]) < sample_num:
                        data[all_value[position_index]].append(input_x)
                else:
                    data[all_value[position_index]] = [input_x]


            hooks = cp_register_hook(model, target_layer_names, pooling_type)

            values = []
            input_samples = []
            for key, input_sample in data.items():
                values.extend([key for i in range(len(input_sample))])
                input_samples.extend(input_sample)
            
            print(len(values), len(input_samples))
            input_tensor = torch.stack(input_samples)
            model_input = input_tensor.to(device)
            model(model_input)

            concept_dataset_dict[position] = {"input_sample_img":input_tensor.numpy(), "concept_label":np.array(values).astype(np.float32), "concept_list": [f"{position}_{i}" for i in range(10)], "concept_input_features": {}}

            for i, layer_name in enumerate(target_layer_names):
                concept_dataset_dict[position]["concept_input_features"][layer_name] = features_out[i]

            for h in hooks:
                h.remove()

            features_out.clear()

    return concept_dataset_dict

def train_cp_for_pvr_task(
    explained_model:torch.nn.Module, 
    pvr_training_dataloader:DataLoader,
    cp_save_path:str,
    target_layer_type:list,
    sample_num:int=200,
    pooling_type:str='mean'):

    if not os.path.exists(cp_save_path):
        os.makedirs(cp_save_path)

    concept_cav_list = []
    position_list = ['a','b','c','d']

    for position in position_list:
        for i in range(10):
            concept_cav_list.append(f"{position}_{i}")

    target_layer_names = []

    for name, layer in explained_model.named_modules():
        for layer_definition in target_layer_type:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in target_layer_names:
                    target_layer_names.append(name)


    # concept_datasets = construct_pvr_concept_dataset_for_cp(
    #     pvr_training_dataloader, 
    #     concept_cav_list, 
    #     position_list, 
    #     sample_num,
    #     explained_model, 
    #     target_layer_names,
    #     pooling_type
    #     )

    # pickle.dump(concept_datasets, open(os.path.join(cp_save_path, f"concept_dataset.txt"), 'wb'))

    concept_datasets = construct_pvr_position_dataset(
        pvr_training_dataloader, 
        position_list, 
        sample_num,
        explained_model, 
        target_layer_names,
        pooling_type
        )

    pickle.dump(concept_datasets, open(os.path.join(cp_save_path, f"concept_position_dataset.txt"), 'wb'))

    best_cps_accuracy = {}

    for concept_name in concept_datasets.keys():
        concept_label = concept_datasets[concept_name]["concept_label"]
        concept_input_features = concept_datasets[concept_name]["concept_input_features"]
        concept_list = concept_datasets[concept_name]["concept_list"]

        cps_for_one_concept = []

        for layer_name, concept_input_feature in concept_input_features.items():
            cp = ConceptRepresentation(
                concept_name, 
                concept_list,
                layer_name, 
                {
                    "model_type":'linear',
                    "alpha":0.01
                    }, 
                save_path=os.path.join(cp_save_path, f"cp_for_{concept_name}_{layer_name}.txt")
            )

            cp.train(concept_input_feature,concept_label)

            cps_for_one_concept.append((cp,cp.accuracies['overall']))

        sorted_cps = sorted(cps_for_one_concept, key = lambda kv:kv[1], reverse=True)

        best_cps_accuracy[concept_name] = (sorted_cps[0][0].accuracies['overall'], sorted_cps[0][0].bottleneck)

        sorted_cps[0][0].save_path = os.path.join(cp_save_path, f"best_cp_for_{concept_name}.txt")

        sorted_cps[0][0].save_cp()

    print("Best CPs Accuracy: ", best_cps_accuracy)

def show_concept_dataset_for_cp(
    cp_save_path:str,
    ):

    concept_dataset = pickle.load(open(os.path.join(cp_save_path,f"concept_position_dataset.txt"), 'rb'))

    for concept_name in concept_dataset.keys():
        model_input = concept_dataset[concept_name]["input_sample_img"]

        plt.subplot(1,3,1)
        plt.imshow(model_input[0].transpose(1,2,0))
        plt.title(concept_name)
        plt.subplot(1,3,2)
        plt.imshow(model_input[-1].transpose(1,2,0))
        plt.subplot(1,3,3)
        plt.imshow(model_input[1].transpose(1,2,0))
        plt.show()


def concept_detection(
    explained_model: torch.nn.Module,
    explained_sample: torch.Tensor,
    cp_save_path: str,
    pooling_type: str='mean'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explained_sample = explained_sample.to(device).unsqueeze(0)

    concept_detection_results = []

    for root, folder, files in os.walk(cp_save_path):
        for file in files:
            if file.split("_")[0] == "best":
                cp = ConceptRepresentation()
                cp.load_from_txt(os.path.join(root,file))

                print(cp.concept_list)

                layer_name = cp.bottleneck

                forward_hook = cp_register_hook(explained_model, [layer_name], pooling_type)
                
                output = torch.softmax(explained_model(explained_sample), dim=1)

                # prediction_index = torch.argmax(output,dim=1)

                cp_input_feature = features_out[0]
                
                pred = int(cp.predict(cp_input_feature)[0])

                # if pred>0:
                concept_detection_results.append(f"{cp.concept_name}_{pred}")

                features_out.clear()
                
                [h.remove() for h in forward_hook]

    print(f"Sample contain concept {concept_detection_results}")

    return concept_detection_results

def calculate_local_concept_sensitivity(
    explained_model: torch.nn.Module,
    explained_sample: torch.Tensor,
    cp_save_path: str,
    pooling_type: str,
    contained_concept: list = None
    ) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explained_sample = explained_sample.to(device).unsqueeze(0)

    original_prediction = torch.softmax(explained_model(explained_sample),dim=1).squeeze()
    original_output_index = torch.argmax(original_prediction).cpu().item()

    concept_sensitivity = []
    # Load CP
    for root, folder, files in os.walk(cp_save_path):
        for file in files:
            if file.split("_")[0] == "best":
                cp = ConceptRepresentation()
                cp.load_from_txt(os.path.join(root,file))

                layer_name = cp.bottleneck

                target_concept = None
                for c in cp.concept_list:
                    if c in contained_concept:
                        target_concept = c
                        break
                
                if target_concept == None:
                    print(f"Input sample do not contain concept {cp.concept_name} with contained_concept: {contained_concept}. Concept {cp.concept_name} contain {cp.concept_list}.")
                    continue

                do_value = torch.from_numpy(cp.get_direction(target_concept)).to(device).unsqueeze(0)
                # add forward_hook
                forward_hook = register_hook_for_do_calculate(explained_model, layer_name, do_value, pooling_type)

                if len(forward_hook) != 1:
                    raise Exception(f"Identify layer name failed {cp.bottleneck} with {len(forward_hook)} identified layer")

                new_output = explained_model(explained_sample)
                new_prediction = torch.softmax(new_output,dim=1).squeeze()

                conceptual_sensitivity = original_prediction - new_prediction

                concept_sensitivity.append((target_concept, layer_name, conceptual_sensitivity[original_output_index].cpu().item()))

                [h.remove() for h in forward_hook]

    concept_sensitivity = sorted(concept_sensitivity, key= lambda kv:kv[2], reverse=True)  
    print(concept_sensitivity)

    return concept_sensitivity, original_output_index

def construct_dataset_for_causality_identification(
    concept_datasets:dict,
    num_samples: int=200,
):  
    '''
    sample_dict:
    {

    }
    '''
    sample_dict = {}

    for position_concept in concept_datasets.keys():
        
        if num_samples <= concept_datasets[position_concept]['input_sample_img'].shape[0]:
            selected_index = random.sample(range(concept_datasets[position_concept]['input_sample_img'].shape[0]), num_samples)
            input_samples = [concept_datasets[position_concept]['input_sample_img'][i] for i in selected_index]
            model_prediction = [concept_datasets[position_concept]['model_prediction'][i] for i in selected_index]

            if not "input_samples" in sample_dict:
                sample_dict["input_samples"] = input_samples
            else:
                sample_dict["input_samples"].extend(input_samples)

            if not "model_prediction" in sample_dict:
                sample_dict["model_prediction"] = model_prediction
            else:
                sample_dict["model_prediction"].extend(model_prediction)

            if not "features_in_layers" in sample_dict:
                sample_dict["features_in_layers"] = {}

            for layer_name in concept_datasets[position_concept]['concept_input_features'].keys():
                features = [concept_datasets[position_concept]['concept_input_features'][layer_name][i] for i in selected_index]
                if not layer_name in sample_dict["features_in_layers"]:
                    sample_dict["features_in_layers"][layer_name] = features
                else:
                    sample_dict["features_in_layers"][layer_name].extend(features)
        else:
            raise Exception(f"Try to select more sample from concept datasets")

    return sample_dict

def chi_square_for_conditional_independence(data, x, y, z:list):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    :math:`P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as :math:`P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    Returns
    -------
    CI Test Results: tuple or bool
        Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27C2 Y | Zs is True.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """

    if hasattr(z, "__iter__"):
        z = list(z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(z)}")

    if (x in z) or (y in z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {x if y in z else y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(z) == 0:
        chi, p_value, dof, expected = chi2_contingency(
            data.groupby([x, y]).size().unstack(y, fill_value=0)
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(z):
            try:
                dataset = df.groupby([x, y]).size().unstack(y, fill_value=0)
                c, pvalue, d, _ = chi2_contingency(
                    df.groupby([x, y]).size().unstack(y, fill_value=0)
                )
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    print(
                        f"Skipping the test {x} \u27C2 {y} | {z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(z, z_state)]
                    )
                    print(
                        f"Skipping the test {x} \u27C2 {y} | {z_str}. Not enough samples"
                    )

        results = stats.chi2.cdf(chi, df=dof)
        print(results)
        p_value = 1 - stats.chi2.cdf(chi, df=dof)  

    # Step 4: Return the values
    return chi, p_value, dof


class CausalGraph:
    def __init__(self) -> None:
        self.graph=Network()

    def add_node(self, node):
        self.graph.add_node(node, size=10)
    
    def add_bi_directional_edge(self, node1, node2):
        if not node1 in self.graph.nodes:
            self.graph.add_node(node1, size=10)
        if not node2 in self.graph.nodes:
            self.graph.add_node(node2, size=10)
        
        self.graph.add_edge(node1, node2)
        self.graph.add_edge(node2, node1)

    def add_directional_edge(self, node1, node2):
        if not node1 in self.graph.nodes:
            self.graph.add_node(node1, size=10)
        if not node2 in self.graph.nodes:
            self.graph.add_node(node2, size=10)
        
        self.graph.add_edge(node1, node2)

    def get_neighbors(self, node=None):
        adj_list = self.graph.get_adj_list()
        if node == None:
            return adj_list
        return adj_list[node]

    def show_graph(self):
        self.graph.show('nx.html', notebook=False)

    def set_direct(self, boolean):
        self.graph.directed = boolean

    def save(self,path):
        save_object = {
            'Edges': self.graph.edges,
            'Nodes': self.graph.node_ids
        }
        pickle.dump(save_object, open(path,'wb'))

    def load(self,path):
        load_object = pickle.load(open(path,'rb'))
        print(load_object)
        for node in load_object['Nodes']:
            self.graph.add_node(node)

        for edge in load_object['Edges']:
            self.graph.add_edge(edge['from'], edge['to'])

    def __repr__(self) -> str:
        return str(self.graph)


def identify_global_concept_causality_graph(
    explained_model: torch.nn.Module,
    cp_save_path: str,
    num_samples: int=200,
    independent_thresholds: float=0.00002
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    concept_datasets = pickle.load(open(os.path.join(cp_save_path, f"concept_position_dataset.txt"), 'rb'))

    for position_concept in concept_datasets.keys():
        model_input = torch.from_numpy(concept_datasets[position_concept]['input_sample_img']).to(device)
        model_output = explained_model(model_input)
        model_prediction = torch.argmax(torch.softmax(model_output, dim=1),dim=1).cpu().numpy()
        concept_datasets[position_concept]['model_prediction'] = model_prediction

    samples_for_causality = construct_dataset_for_causality_identification(
        concept_datasets,
        num_samples,
        )

    concept_list = []

    for root, folder, files in os.walk(cp_save_path):
        for file in files:
            if file.split("_")[0] == "best":
                cp = ConceptRepresentation()
                cp.load_from_txt(os.path.join(root,file))
                layer_name = cp.bottleneck

                input_features = samples_for_causality["features_in_layers"][layer_name]
                concept_index = cp.predict(input_features)
                if not 'concept_prediction' in samples_for_causality:
                    samples_for_causality['concept_prediction'] = {}
                samples_for_causality['concept_prediction'][cp.concept_name] = concept_index

                concept_list.append(cp.concept_name)
    samples_for_causality['concept_prediction']['model_output'] = samples_for_causality['model_prediction']

    # Init graph structure
    init_graph = CausalGraph()
    for c in concept_list:
        init_graph.add_directional_edge("Model Input", c)

    concept_list.append('model_output')

    # Calculate independent value for adding skeletons
    for i, c in enumerate(concept_list):
        for cc in concept_list[i+1:]:
            samples_for_causality['concept_prediction'][c]
            samples_for_causality['concept_prediction'][cc]
            table = pd.crosstab(samples_for_causality['concept_prediction'][c], samples_for_causality['concept_prediction'][cc])
            independent_results = chi2_contingency(table)
            print(independent_results)

            if independent_results.pvalue < independent_thresholds:
                print(c,cc,'is correlation, p value: ', independent_results.pvalue)
                if cc=='model_output':
                    init_graph.add_directional_edge(c, cc)
                elif c=='model_output':
                    init_graph.add_directional_edge(cc, c)
                else:
                    init_graph.add_bi_directional_edge(c,cc)
            else:
                print(c,cc,'is independent, p value: ', independent_results.pvalue)

    # init_graph.show_graph()
                
    related_concept_list = init_graph.get_neighbors('model_output')

    contained_concept_list = init_graph.get_neighbors('Model Input')

    non_related_concept_list = contained_concept_list - related_concept_list

    print(f'related_concept_list {related_concept_list}, contained_concept_list {contained_concept_list}, non_related_concept_list {non_related_concept_list}')

    for i, c in enumerate(non_related_concept_list):
        for cc in related_concept_list:
            dataset = pd.DataFrame.from_dict(samples_for_causality['concept_prediction'])
            chi, p_value, dof = chi_square_for_conditional_independence(dataset, c, cc, ['model_output'])

            if p_value < independent_thresholds:
                print(f'{c}, and model_output is correlation on given {cc}, p value: {p_value}')
                init_graph.add_directional_edge(c,cc)
            else:
                print(f'{c}, and model_output is conditional independent on given {cc}, p value: {p_value}')

    # init_graph.show_graph()
    # Finish adding skeletons
    # Start validating causality
    init_graph.set_direct(True)

    init_graph.save(os.path.join(cp_save_path, f"global_graph.txt"))

    init_graph.load(os.path.join(cp_save_path, f"global_graph.txt"))

    init_graph.show_graph()



def calculate_local_concept_sensitivity_based_on_gradient(
        explained_model: torch.nn.Module,
        explained_sample: torch.Tensor,
        cp_save_path: str,
        pooling_type: str,
        contained_concept: list = None
    ) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explained_sample = explained_sample.to(device).unsqueeze(0)

    concept_representations = []
    layer_list = []
    # Load CP
    for root, folder, files in os.walk(cp_save_path):
        for file in files:
            if file.split("_")[0] == "best":
                cp = ConceptRepresentation()
                cp.load_from_txt(os.path.join(root,file))

                layer_name = cp.bottleneck

                target_concept = None
                for c in cp.concept_list:
                    if c in contained_concept:
                        target_concept = c
                        break
                
                if target_concept == None:
                    print(f"Input sample do not contain concept {cp.concept_name} with contained_concept: {contained_concept}. Concept {cp.concept_name} contain {cp.concept_list}.")
                    continue

                c = torch.from_numpy(cp.get_direction(target_concept)).to(device)

                concept_representations.append((target_concept, cp.bottleneck, c))

                if not layer_name in layer_list:
                    layer_list.append(layer_name)

                    # add forward_hook
    backward_hook = register_gradient_hook_for_cq(
                        explained_model, 
                        layer_list)

    output = explained_model(explained_sample)
    new_prediction = torch.softmax(output,dim=1).squeeze()
    predict_index = torch.argmax(new_prediction)
    # Backward process
    one_hot = F.one_hot(predict_index,num_classes=new_prediction.size()[-1])
    one_hot = one_hot.float().requires_grad_(True)
    one_hot = torch.sum(one_hot * output) 
    explained_model.zero_grad()
    one_hot.backward(retain_graph=True)

    grads = {}
    for index, item in enumerate(backward_hook):
        grads[item[0]] = grad_out[-1-index]

    concept_sensitivity = []
    for item in concept_representations:
        target_concept, layer_name, c = item[0], item[1], item[2]
        if pooling_type == "mean":
            grad = torch.mean(grads[layer_name].reshape([grads[layer_name].shape[0],grads[layer_name].shape[1],-1]), dim=-1).squeeze()
        elif pooling_type == "max":
            # grad = grads[cp.bottleneck]
            grad = torch.mean(grads[layer_name].reshape([grads[layer_name].shape[0],grads[layer_name].shape[1],-1]), dim=-1).squeeze()
        sensitive = torch.dot(c, grad).cpu().item()
        concept_sensitivity.append((target_concept, layer_name, sensitive))

    grad_out.clear()
    [item[1].remove() for item in backward_hook]

    concept_sensitivity = sorted(concept_sensitivity, key= lambda kv:kv[2], reverse=True)  

    print(concept_sensitivity)
    return concept_sensitivity, predict_index

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





                


