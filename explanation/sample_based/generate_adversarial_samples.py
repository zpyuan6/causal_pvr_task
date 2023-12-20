import torch
import torch.nn as nn

def g(old_predict,new_predict_label):
    # print("g",torch.argmax(old_predict),old_predict,torch.argmax(new_predict_label),new_predict_label)
    if  torch.argmax(old_predict)!=torch.argmax(new_predict_label):
        return True
    return False

# merge mask, clips elements of its argument to a valid input range
def merge_mask(img_data,mask):
    new_img = img_data+mask
    # print(f"merge_mask {img_data},\n{mask},\n{img_data.shape},{mask.shape}")
    max_value = torch.max(new_img)
    min_value = torch.min(new_img)
    new_img = (new_img-min_value)/(max_value-min_value)
    
    return new_img

def optimizer_SGD(mask, delta_mask):
    learning_rate = 1

    return mask-learning_rate*delta_mask

# Adversarial explanations for understanding image classification decisions and improved neural network robustness
def gen_adversarial_sample(num_step:int,model:nn.Module,img_data,device,target_label:int,input_size):
    # do one prediction
    model.to(device)
    model.eval()
    img_data = img_data.to(device)
    target_label = torch.Tensor([target_label]).type(torch.LongTensor).to(device)
    output = model(img_data)
    loss_function = nn.CrossEntropyLoss().to(device)

    best_mask = mask = torch.zeros(input_size)
    # norm l2 of mask
    best_mask_lt = float("inf")

    for i in range(num_step):
        # print("===========================================")
        mask = mask.to(device).requires_grad_(True)
        x = merge_mask(img_data,mask).to(device)
        y = model(x)

        if g(output,y):
            # print("result change: ",output, y, torch.argmax(output),torch.argmax(y))
            mask_lt = torch.norm(mask,p=2)/mask.numel()
            # print(f"mask.numel() {mask.numel()}")
            mean_mask = torch.mean(mask)
            # print("mean_mask: ",mean_mask)
            dir_delta_mask = mean_mask/abs(mean_mask)
            delta_mask = mask_lt*dir_delta_mask

            if mask_lt<best_mask_lt:
                best_mask_lt = mask_lt
                best_mask = mask
        else:
            loss_value = loss_function(y,target_label)
            model.zero_grad()
            loss_value.backward()
            mask_grad = mask.grad.data
            # print("output == y, mask_grad",mask_grad)
            # delta_mask = mask_grad / torch.norm(mask_grad,p=2)
            delta_mask = mask_grad

        with torch.no_grad():
            mask = optimizer_SGD(mask, delta_mask)

    return best_mask, merge_mask(img_data,best_mask)




        
