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
# From attribution maps to humanunderstandable explanations through Concept Relevance Propagation

def conditional_attributions(
    model:nn.Module, 
    data_sample:torch.tensor, 
    conditional_layer_name:str,
    conditional_channel:list, 
    conditional_output:list
    ):
    # https://github.com/rachtibat/zennit-crp/blob/master/tutorials/attributions.ipynb

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    # load CRP toolbox
    attribution = CondAttribution(model)

    # here, each channel is defined as a concept
    # or define your own notion!
    cc = ChannelConcept()

    # get layer names of Conv2D and MLP layers
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])

    # get a conditional attribution for target_channel 50 in layer layer_name wrt. output target_output
    conditions = [{conditional_layer_name: conditional_channel, 'y': conditional_output}]

    attr = attribution(data_sample, conditions, composite, record_layer=layer_names)

    # heatmap and prediction
    heatmap = imgify(attr.heatmap, symmetric=True)
    heatmap.show()
    print(f"prediction: {attr.prediction}")
    # attr.prediction attr.activations
    # activations and relevances for each layer name
    # attr.relevances

    # relative importance of each concept for final prediction
    rel_c = cc.attribute(attr.relevances[conditional_layer_name])
    # most relevant channels in features.40
    concept_ids = torch.argsort(rel_c, descending=True)[0, :5]

    print(concept_ids)

    return concept_ids.cpu()


def feature_visualization(
    model:nn.Module, 
    data_set, 
    concept_ids:list,
    layer_name:str
    ):

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()

    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    layer_map = {layer : cc for layer in layer_names}

    attribution = CondAttribution(model)

    # compute visualization (it takes for VGG16 and ImageNet testset on Titan RTX 30 min)
    fv = FeatureVisualization(attribution, data_set, layer_map)
    fv.run(composite, 0, len(data_set), 32, 100)

    # visualize MaxRelevance reference images for top-5 concepts
    ref_c = fv.get_max_reference(concept_ids, layer_name, 'relevance', (0,5), composite=composite)

    plot_grid(ref_c)