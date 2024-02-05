from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.image import imgify
from PIL import Image
from crp.graph import trace_model_graph
from crp.visualization import FeatureVisualization
from crp.image import plot_grid
import torch.nn as nn
import torch
from typing import List

import matplotlib.pyplot as plt

from torchvision.transforms.functional import gaussian_blur
from crp.helper import max_norm

from crp.image import vis_opaque_img, get_crop_range
from crp.attribution import AttributionGraph

import plotly.graph_objects as go

# From attribution maps to humanunderstandable explanations through Concept Relevance Propagation

def identify_graph(
    model:nn.Module, 
    data_sample:torch.tensor,
    data_set,
    conditional_layer_name:str,
    show_samples: bool = False
    ):

    data_sample = data_sample.unsqueeze(0)
    data_sample.requires_grad = True

    output = model(data_sample)
    predict = torch.softmax(output, 1)
    target_y = torch.argmax(predict[0]).cpu().item()

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    attribution = CondAttribution(model, no_param_grad=True)



    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    # layer_names = get_layer_names(model, [torch.nn.BatchNorm2d])

    graph = trace_model_graph(model, data_sample, layer_names)
    # print(graph)

    cc = ChannelConcept()
    conditions = [{'y': [target_y]}]

    attr = attribution(data_sample, conditions, composite, record_layer=layer_names)
    rel_c = cc.attribute(attr.relevances[conditional_layer_name])

    concept_ids = torch.argsort(rel_c, descending=True)[0, :5]
    print(f"Concept ids {concept_ids}, with rel {torch.take(rel_c, concept_ids) }")
    concept_id = concept_ids.cpu().tolist()[0]

    layer_map = {name: cc for name in layer_names}
    attgraph = AttributionGraph(attribution, graph, layer_map)

    nodes, connections = attgraph(data_sample, composite, concept_id, conditional_layer_name, target_y, width=[1, 1], abs_norm=True)
    print("\nNodes:\n", nodes)
    print("\nConnections:\n", connections)

    channel_dict = {}

    for item in nodes:
        layer, ch = item
        if layer in channel_dict:
            channel_dict[layer].append(ch)
        else:
            channel_dict[layer] = [ch]

    print(channel_dict)

    node_list = [f"Output: {target_y}"]
    node_list.extend([node[0] for node in nodes])
    node_labels = [f"Output: {target_y}"]
    node_labels.extend([f"{node[0]} Channel {node[1]}".replace("layer", "") for node in nodes])

    # color = ['rgba(255,0,255, 0.8)' for i in len(node_labels)]
    # line_color = [item.replace("0.8", "0.2")  for item in color]

    source_list = [1]
    target_list = [0]
    value_list  = [rel_c[0][concept_id].cpu().item()]

    for target in connections.keys():
        target_index = node_list.index(target[0])
        
        for source in connections[target]:
            target_list.append(target_index)
            source_index = node_list.index(source[0])
            source_list.append(source_index)
            value_list.append(source[2])


    fig = go.Figure(
        data=[
                go.Sankey(
                    node = dict(
                        pad = 5,
                        label = node_labels,
                    ),
                    link=dict(
                        arrowlen=60,
                        source = source_list,
                        target = target_list,
                        value = value_list,
                    ),
                    textfont=dict(
                        size=14,
                        color='black'
                    )
                )
            ]
        )

    fig.show()

    if show_samples:
        fv = FeatureVisualization(attribution, data_set, layer_map)
        fv.run(composite, 0, len(data_set), 32, 100)

        for item in channel_dict.items():
            layer_name, concept_ids = item
            ref_c = fv.get_max_reference(concept_ids, layer_name, 'relevance', (0,3), composite=composite)

            plot_grid(ref_c, figsize=(5*3, len(concept_ids)*5))

def conditional_attributions(
    model:nn.Module, 
    data_sample:torch.tensor, 
    conditional_layer_name:str
    ):
    # https://github.com/rachtibat/zennit-crp/blob/master/tutorials/attributions.ipynb

    data_sample = data_sample.unsqueeze(0)
    data_sample.requires_grad = True

    output = model(data_sample)
    predict = torch.softmax(output, 1)
    class_idx = torch.argmax(predict[0]).cpu().item()

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    # load CRP toolbox
    attribution = CondAttribution(model)

    # get layer names of Conv2D and MLP layers
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])

    # get a conditional attribution for target_channel 50 in layer layer_name wrt. output target_output
    # conditions = [{conditional_layer_name: [5], 'y': [class_idx]}]
    conditions = [{'y': [class_idx]}]

    attr = attribution(data_sample, conditions, composite, record_layer=layer_names)

    # heatmap and prediction
    # heatmap = imgify(attr.heatmap, symmetric=True)
    # heatmap.show()
    print(f"prediction: {attr.prediction} {class_idx}")
    # attr.prediction attr.activations
    # activations and relevances for each layer name
    # attr.relevances

    # relative importance of each concept for final prediction
    # here, each channel is defined as a concept
    # or define your own notion!
    cc = ChannelConcept()
    rel_c = cc.attribute(attr.relevances[conditional_layer_name])
    # most relevant channels in features.40
    concept_ids = torch.argsort(rel_c, descending=True)[0, :5]
    print(f"Concept ids {concept_ids}, with rel {torch.take(rel_c, concept_ids) }")
    concept_ids = concept_ids.cpu().tolist()


    conditions_with_target_concept = []
    conditions_with_target_concept.append({'y': [class_idx]})
    conditions_with_target_concept.extend([{conditional_layer_name:[concept_id], 'y': [class_idx]} for concept_id in concept_ids])
    
    print(conditions_with_target_concept)

    attr_with_target_concept = attribution(data_sample, conditions_with_target_concept, composite, record_layer=layer_names)

    print("Heatmap mean", torch.mean(torch.reshape(attr_with_target_concept.heatmap,(attr_with_target_concept.heatmap.shape[0],-1)), dim=1) )

    heatmap = imgify(attr_with_target_concept.heatmap, symmetric=True, grid=(1, len(concept_ids)+1))
    heatmap.show()


def feature_visualization(
    model:nn.Module, 
    data_set, 
    data_sample:torch.tensor,
    layer_name:str
    ):

    data_sample = data_sample.unsqueeze(0)
    data_sample.requires_grad = True

    output = model(data_sample)
    predict = torch.softmax(output, 1)
    class_idx = torch.argmax(predict[0]).cpu().item()

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    cc = ChannelConcept()

    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    layer_map = {layer : cc for layer in layer_names}

    attribution = CondAttribution(model)

    conditions = [{'y': [class_idx]}]

    attr = attribution(data_sample, conditions, composite, record_layer=layer_names)

    # relative importance of each concept for final prediction
    rel_c = cc.attribute(attr.relevances[layer_name])
    # most relevant channels in features.40
    concept_ids = torch.argsort(rel_c, descending=True)[0, :4]

    print(f"Concept ids {concept_ids}, with rel {torch.take(rel_c, concept_ids) }")
    concept_ids = concept_ids.cpu().tolist()

    # compute visualization (it takes for VGG16 and ImageNet testset on Titan RTX 30 min)
    fv = FeatureVisualization(attribution, data_set, layer_map)
    fv.run(composite, 0, len(data_set), 32, 100)

    # visualize MaxRelevance reference images for top-5 concepts
    # ref_c = fv.get_max_reference(concept_ids, layer_name, 'relevance', (0,5), composite=composite)
    ref_c = fv.get_max_reference(concept_ids, layer_name, 'relevance', (0,5), composite=composite, plot_fn=vis_opaque_img)
    # ref_c = fv.get_max_reference(concept_ids, layer_name, 'relevance', (0,5), composite=composite, plot_fn=vis_hot_img)

    plot_grid(ref_c, figsize=(6,5))


@torch.no_grad()
def vis_hot_img(data_batch, heatmaps, rf=False, alpha=0.3, vis_th=0.2, crop_th=0.1, kernel_size=19) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th. 
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = max_norm(gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0])
        vis_mask = filtered_heat > vis_th 
       
        if rf:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            
            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t
        
        # print(type(img), img.shape, type(vis_mask),vis_mask.shape)
        img = img.permute(1,2,0)

        fig = plt.Figure(figsize=img.shape[:2])

        plt.imshow(img)
        plt.imshow(vis_mask, alpha=0.5)
        plt.axis('off')
        fig = plt.figure()

        img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

        # inv_mask = ~vis_mask
        # img = img * vis_mask + img * inv_mask * alpha
        # img = zimage.imgify(img.detach().cpu())

        imgs.append(img)

    return imgs