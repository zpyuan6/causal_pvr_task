# Grad-Cam++
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

def cam_grad_pp_register_hook(model:nn.Module, layer_name:str):
    def forward_hook(module, input, output):
        print("forward_hook")
        features_in_hook.append(input)
        features_out_hook.append(output)
        return None

    def backward_hook(module, grad_input, grad_output):
        print("backward_hook")
        grad_in_hook.append(grad_input)
        grad_out_hook.append(grad_output)
        return None

    for name, module in model.named_modules():
            # print(name)
        if name == layer_name:
            print("find layer ", name)
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

def do_prediction_with_grad(model,device,img_path,input_shape):

    try:
        print(f"Try to open image {img_path}")
        image = Image.open(img_path).convert('RGB')
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

def generate_grad_cam_pp(last_layer_name:str, model:nn.Module, input_size, img_path, device):
    cam_grad_pp_register_hook(model, last_layer_name)
    class_idx,predict,image_data = do_prediction_with_grad(model,device,img_path,input_size)

    # 利用onehot的形式锁定目标类别
    one_hot = F.one_hot(class_idx,num_classes=predict.size()[-1])
    one_hot = one_hot.float().requires_grad_(True)
    #  获取目标类别的输出,该值带有梯度链接关系,可进行求导操作, sum 为什么？
    one_hot = torch.sum(one_hot * predict) 

    model.zero_grad()
    # backward 求导 获取特征权重的过程
    one_hot.backward(retain_graph=True)

    # 获取对应特征层的梯度map
    grads_val = grad_out_hook[-1][0].cpu()
    # 注意,这里用乘法,因为论文中将2次偏导和3次偏导进行了2次方和3次方的转化 二次偏导和三次偏导转化为幂次方？为什么？
    grad_2 = grads_val.pow(2)
    grad_3 = grads_val.pow(3)
    features_map = features_out_hook[-1].cpu()

    print(grads_val.shape)
    print(grad_2.shape)
    print(grad_3.shape)

    c =  (grad_3 * features_map).sum(axis=(2, 3), keepdims=True)
    # 获取alpha权重map
    alpha = (grad_2 / (2 * grad_2 + c)).squeeze()
    # 利用alpha 权重map去获取特征权重
    alpha = alpha.mul_(torch.relu(grads_val.squeeze(0))).sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)

    features_map = features_map.squeeze()
    cam = (alpha*features_map).sum(dim=0)
    print("cam: ",cam.shape)

        # relu操作,去除负值
    cam = F.relu(cam, inplace=True)
    # 归一化操作, 并缩放到原图尺寸
    batch_cams = F.normalize(cam)
    pil_image = transforms.ToPILImage()(batch_cams)
    cam = pil_image.resize((input_size[0],input_size[1]))

    return image_data,cam