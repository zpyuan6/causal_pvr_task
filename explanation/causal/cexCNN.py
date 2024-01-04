import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import copy

def filter_importance(model:nn.Module, input_sample:torch.tensor, layer_name:str):
    input_img = torch.unsqueeze(input_sample, 0)
    prediction = model(input_img)
    orignail_acc = nn.Softmax(dim=1)(prediction)
    output_index = torch.argmax(orignail_acc, dim=1).squeeze()

    channel_length = model.state_dict()[layer_name+".weight"].shape[1]

    weights_dic = {}
    with tqdm(total=channel_length) as tbar:
        for channel_index in range(channel_length):
            tbar.update(1)
            hooks = set_hook_to_delete_channels(model, layer_name, channel_index)

            new_prediction = model(input_img)
            new_acc = nn.Softmax(dim=1)(new_prediction).squeeze().cpu()
            diff = orignail_acc[0][output_index]-new_acc[output_index]
            weights_dic[channel_index] = [new_acc, new_acc[output_index], diff, abs(diff)]

            [hook.remove() for hook in hooks]

    weights_dic_sort = sorted(weights_dic.items(), key=lambda kv:kv[1][3].item(), reverse=True)

    importance = []
    for item in weights_dic_sort:
        importance.append([item[0], item[1][2].item(), item[1][3].item()])

    return weights_dic, weights_dic_sort, importance


feature_map_for_channel = []

def set_hook_to_delete_channels(model, layer_name, channel_index):

    def forward_hook(module, input, output):
        # print("forward_hook")
        feature_map_for_channel.append(output[:,channel_index,:,:].clone())
        output[:,channel_index,:,:] = 0
        return output
    
    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            # print("find layer ", name)
            hooks.append(module.register_forward_hook(forward_hook))

    return hooks
    

def cexCNN_heatmap(model:nn.Module, input_sample:torch.tensor, layer_name:str):
    importance = filter_importance(model, input_sample, layer_name)[2]

    important_channel_index = importance[0][0]

    weight_for_channel = [0 for i in range(len(importance))]
    for item in importance:
        weight_for_channel[item[0]] = item[1]

    weighted_map = torch.sum(torch.stack(feature_map_for_channel).cpu() * torch.FloatTensor(weight_for_channel).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=0).squeeze()
    important_map = feature_map_for_channel[important_channel_index].squeeze()

    # relu操作,去除负值
    weighted_map = F.relu(weighted_map, inplace=True)
    # 归一化操作, 并缩放到原图尺寸
    weighted_map -= torch.min(weighted_map)
    weighted_map = weighted_map/torch.max(weighted_map)
    pil_image = transforms.ToPILImage()(weighted_map)
    weighted_map = pil_image.resize((input_sample.shape[-1],input_sample.shape[-2]))

    # relu操作,去除负值
    important_map = F.relu(important_map, inplace=True)
    # 归一化操作, 并缩放到原图尺寸
    important_map -= torch.min(important_map)
    important_map = important_map/torch.max(important_map)
    pil_image = transforms.ToPILImage()(important_map)
    important_map = pil_image.resize((input_sample.shape[-1],input_sample.shape[-2]))

    feature_map_for_channel.clear()

    return weighted_map, important_map, importance