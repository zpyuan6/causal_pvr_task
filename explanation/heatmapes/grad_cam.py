import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import numpy as np

features_in_hook = []
features_out_hook = []
grad_in_hook = []
grad_out_hook = []

def cam_grad_register_hook(model:nn.Module, layer_name:str):

    def forward_hook(module, input, output):
        # print("forward_hook")
        features_in_hook.append(input)
        features_out_hook.append(output)
        return None

    def backward_hook(module, grad_input, grad_output):
        # print("backward_hook")
        grad_in_hook.append(grad_input)
        grad_out_hook.append(grad_output)
        return None
    
    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            # print("find layer ", name)
            hooks.append(module.register_forward_hook(forward_hook))
            hooks.append(module.register_backward_hook(backward_hook))

    return hooks

def do_prediction_with_grad(model,device,img_path,input_shape):

    try:
        print(f"Try to open image {img_path}")
        image = Image.open(img_path).convert('L')
        # image = Image.open(img_path).convert('RGB')
    except:
        print('Open Error! Try again!')
    else:
        # image_shape = np.array(np.shape(image)[0:2])
        image = image.resize((input_shape[0], input_shape[1]), Image.BICUBIC)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_data = transform(image)
        # plt.imshow(np.transpose(image_data,(1,2,0)))
        # plt.show()
        image_data = np.expand_dims(image_data, 0)

        
        images = torch.from_numpy(image_data)
        images = images.to(device)
        model.to(device)
        output = model(images)
        preds = torch.softmax(output, 1)
        result = torch.argmax(preds[0])

        return result, output, image_data


def generate_grad_cam(last_layer_name:str, model:nn.Module, input_size, img_path, device):
    hooks = cam_grad_register_hook(model, last_layer_name)
    class_idx,predict,image_data = do_prediction_with_grad(model,device,img_path,input_size)

    # 利用onehot的形式锁定目标类别
    one_hot = F.one_hot(class_idx,num_classes=predict.size()[-1])
    one_hot = one_hot.float().requires_grad_(True)
    #  获取目标类别的输出,该值带有梯度链接关系,可进行求导操作, sum 为什么？
    one_hot = torch.sum(one_hot * predict) 

    model.zero_grad()
    # backward 求导
    one_hot.backward(retain_graph=True)

    # 获取对应特征层的梯度map
    grads_val = grad_out_hook[-1][0].cpu()

    # 获取目标特征输出
    target = features_out_hook[-1].cpu()
    weights = nn.AvgPool2d(grads_val.shape[-1])(grads_val) # 利用GAP操作, 获取特征权重
    target = target.squeeze()
    weights = weights.squeeze().unsqueeze(-1).unsqueeze(-1)
    cam = (weights*target).sum(dim=0)

    # relu操作,去除负值
    cam = F.relu(cam, inplace=True)
    # 归一化操作, 并缩放到原图尺寸
    batch_cams = F.normalize(cam)
    pil_image = transforms.ToPILImage()(batch_cams)
    cam = pil_image.resize((input_size[0],input_size[1]))

    for h in hooks:
        h.remove()

    return image_data,cam



def generate_grad_cam_from_img(last_layer_name:str, model:nn.Module, img:torch.Tensor):
    """
    Input parameter:
        last_layer_name:str 

    Output variable:

    """

    hooks = cam_grad_register_hook(model, last_layer_name)

    input_img = torch.unsqueeze(img, 0)

    output = model(input_img)
    predict = torch.softmax(output, 1)
    class_idx = torch.argmax(predict[0])


    # 利用onehot的形式锁定目标类别
    one_hot = F.one_hot(class_idx,num_classes=predict.size()[-1])
    one_hot = one_hot.float().requires_grad_(True)
    #  获取目标类别的输出,该值带有梯度链接关系,可进行求导操作, sum 为什么？
    one_hot = torch.sum(one_hot * predict) 

    model.zero_grad()
    # backward 求导
    one_hot.backward(retain_graph=True)

    # 获取对应特征层的梯度map
    grads_val = grad_out_hook[-1][0].cpu()

    # 获取目标特征输出
    target = features_out_hook[-1].cpu()
    weights = nn.AvgPool2d(grads_val.shape[-1])(grads_val) # 利用GAP操作, 获取特征权重
    target = target.squeeze()
    weights = weights.squeeze().unsqueeze(-1).unsqueeze(-1)
    cam = (weights*target).sum(dim=0)

    # relu操作,去除负值
    cam = F.relu(cam, inplace=True)
    # 归一化操作, 并缩放到原图尺寸
    cam -= torch.min(cam)
    batch_cams = cam/torch.max(cam)
    # batch_cams = F.normalize(cam)
    pil_image = transforms.ToPILImage()(batch_cams)
    cam = pil_image.resize((img.shape[-1],img.shape[-2]))

    for h in hooks:
        h.remove()

    features_in_hook.clear()
    features_out_hook.clear()
    grad_in_hook.clear()
    grad_out_hook.clear()

    return img.cpu().numpy(), cam