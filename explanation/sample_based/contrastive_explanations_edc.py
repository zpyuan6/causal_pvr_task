# Reference EDC
# Explainable image classification with evidence counterfactual
from skimage.segmentation import quickshift
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
import cv2
from time import time

def sedc_t2_fast(image, classifier, segments, target_class, mode, max_time=600):

    init_time = time()

    input_img = transforms.ToTensor()(image)
    np_img = np.array(image)
    np_img = np_img[...,np.newaxis]
    
    result = classifier(input_img.unsqueeze(0))
    result = torch.softmax(result,1)

    c = torch.argmax(result)
    p = result[0, target_class].detach().numpy()

    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for target class
    sets_to_expand_on = []
    P_sets_to_expand_on = np.array([])

    if mode == 'mean':
        perturbed_image = np.zeros((28,28,1))
        perturbed_image[:,:,0] = np.mean(np_img[:,:,0])
    elif mode == 'blur':
        perturbed_image = cv2.GaussianBlur(image, (3,3), 0)
    elif mode == 'random':
        perturbed_image = np.random.random((28,28,1))
    elif mode == 'inpaint':
        perturbed_image = np.zeros((28,28,1))
        for j in np.unique(segments):
            image_absolute = (image*255).astype('uint8')
            mask = np.full([image_absolute.shape[0],image_absolute.shape[1]],0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            perturbed_image[segments == j] = image_segment_inpainted[segments == j]/255.0

    cf_candidates = []
    for j in np.unique(segments):
        test_image = np_img.copy()
        test_image[segments == j] = perturbed_image[segments == j]

        # plt.imshow(test_image[:,:,0])
        # plt.title(result)
        # plt.show()
        new_tensor_img = transforms.ToTensor()(test_image)

        cf_candidates.append(new_tensor_img)

    cf_candidates = torch.stack(cf_candidates,0)

    results = torch.softmax(classifier(cf_candidates),1)
    results = results.detach().numpy()
    c_new_list = np.argmax(results, axis=1)
    print(f"c_new_list{c_new_list},{results}")
    p_new_list = results[:, target_class]
    print(f"p_new_list{p_new_list},{target_class}")

    if target_class in c_new_list:
        R = [[x] for x in np.where(c_new_list == target_class)[0]]

        target_class_idxs = np.array(R).reshape(1, -1)[0]

        I = cf_candidates[target_class_idxs]
        C = c_new_list[target_class_idxs]
        P = p_new_list[target_class_idxs]

    print(f"np.where(c_new_list != target_class)[0]: {np.where(c_new_list != target_class)[0]}")
    sets_to_expand_on = [[x] for x in np.where(c_new_list != target_class)[0]]

    #目标分类confidence - 原始分类confidence
    P_sets_to_expand_on = p_new_list[np.where(c_new_list != target_class)[0]]-results[np.where(c_new_list != target_class)[0], c]

    combo_set = [0]
    
    while len(R) == 0 and len(combo_set) > 0 and max_time > time() - init_time:
        print(f"==============================================================")
        print(f"sets_to_expand_on{sets_to_expand_on}")
        print(f"P_sets_to_expand_on{P_sets_to_expand_on}")
        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        print(f"combo,{combo}")
        print(f"np.unique(segments) {np.unique(segments)}")
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on, combo)

        cf_candidates = []

        print(f"combo_set,{combo_set}")
        for cs in combo_set:
            test_image = np_img.copy()
            for k in cs:
                test_image[segments == k] = perturbed_image[segments == k]

            new_tensor_img = transforms.ToTensor()(test_image)
            cf_candidates.append(new_tensor_img)

        if len(cf_candidates)==0:
            break

        cf_candidates = torch.stack(cf_candidates,0)

        results = torch.softmax(classifier(cf_candidates),1)
        results = results.detach().numpy()
        c_new_list = np.argmax(results, axis=1)
        p_new_list = results[:, target_class]

        if target_class in c_new_list:
            selected_idx = np.where(c_new_list == target_class)[0]

            R = np.array(combo_set)[selected_idx].tolist()
            I = cf_candidates[selected_idx]
            C = c_new_list[selected_idx]
            P = p_new_list[selected_idx]

        sets_to_expand_on += np.array(combo_set)[np.where(c_new_list != target_class)[0]].tolist()
        P_sets_to_expand_on = np.append(P_sets_to_expand_on, p_new_list[np.where(c_new_list != target_class)[0]] - results[np.where(c_new_list != target_class)[0], c])

    # Select best explanation: highest target score increase

    if len(R) > 0:
        print(type(P),type(p))
        best_explanation = np.argmax(P - p)
        segments_in_explanation = R[best_explanation]
        explanation = np.full([28,28,1],0/255.0)
        for i in R[best_explanation]:
            explanation[segments == i] = np_img[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]

        return explanation, segments_in_explanation, perturbation, new_class

    print('No CF found on the requested parameters')
    return None, None, None, c



def sedc_t2_mnist(device, image, classifier, segments, target_class, mode):

    max_time=15
    init_time = time()

    segments = segments[np.newaxis,...]
    classifier.to(device=device)

    # print(image.shape,"!!!!!!!!!!!!")
    result = classifier(torch.tensor(image[np.newaxis,...]).to(device=device))
    result = torch.softmax(result,1).detach().numpy()
    orig_result = result
    c = np.argmax(result)
    # p = result[0,target_class] #confidence of target class
    p = result[0,c]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for target class
    sets_to_expand_on = [] #send to expand segmentation image
    P_sets_to_expand_on = np.array([])

    if mode == 'mean':
        perturbed_image = np.zeros((1,28,28))
        perturbed_image[0,:,:] = np.mean(image[0,:,:])
    elif mode == 'blur':
        perturbed_image = cv2.GaussianBlur(image, (3,3), 0)
    elif mode == 'random':
        perturbed_image = np.random.random((1,28,28))
    elif mode == 'black':
        perturbed_image = np.zeros((1,28,28))
        perturbed_image[0,:,:] = 0
    elif mode == 'white':
        perturbed_image = np.zeros((1,28,28))
        perturbed_image[0,:,:] = 1
    
    # print(np.unique(segments))
    for j in np.unique(segments):
        test_image = image.copy()
        test_image[segments == j] = perturbed_image[segments == j]

        # plt.imshow(test_image.squeeze())
        # plt.show()

        result = classifier(torch.tensor(test_image[np.newaxis,...]))
        result = torch.softmax(result,1).detach().numpy()
        c_new = np.argmax(result)
        # p_new = result[0,target_class]
        p_new = result[0,c_new]
        
        if c_new == target_class:
        # if c_new != c:
            R.append([j])
            I.append(test_image)
            C.append(c_new)
            P.append(p_new)
        else: 
            sets_to_expand_on.append([j])
            #confidence of target class minus confidence of original prediction class 
            P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new-result[0,c]) 
    
    # print(np.unique(segments), len(R), len(P_sets_to_expand_on))
    # while len(R) == 0 and max_time > time() - init_time and len(P_sets_to_expand_on)!=0:
    while len(R) == 0:
        # print("---------------------------------")
        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # print(combo)
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on,combo)
        
        for cs in combo_set: 
            
            test_image = image.copy()
            for k in cs: 
                test_image[segments == k] = perturbed_image[segments == k]
                
            result = classifier(torch.tensor(test_image[np.newaxis,...]).to(device=device))
            result = torch.softmax(result,1).detach().numpy()
            c_new = np.argmax(result)
            # p_new = result[0,target_class]
            p_new = result[0,c_new]
                
            if c_new == target_class:
            # if c_new != c:
                R.append(cs)
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)
            else: 
                sets_to_expand_on.append(cs)
                P_sets_to_expand_on = np.append(P_sets_to_expand_on,p_new-result[0,c])
              
    # Select best explanation: highest target score increase
    
    if len(R) > 0:

        best_explanation = np.argmax(P - p) 
        segments_in_explanation = R[best_explanation]
        explanation = np.full([image.shape[0],image.shape[1],image.shape[2]],0/255.0)
        for i in R[best_explanation]:
            explanation[segments == i] = image[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]        
                    
        return explanation, segments_in_explanation, perturbation, new_class, P[best_explanation], orig_result, result 

    print("Not find counterfactual")
    return None, None, image, torch.tensor(0), torch.tensor(0), orig_result, result           



def find_cf_region(device, model, image_data, target_label):
    cf_mode = ['mean', 'blur', 'random', 'inpaint'][2]
    # cf_timeout = 60

    pil_img = transforms.ToPILImage()(image_data)

    segments = quickshift(pil_img.convert("RGB"), kernel_size=1, max_dist=15, ratio=0.2)
    # segments = quickshift(pil_img.convert("RGB"), kernel_size=2, max_dist=50, ratio=0.2)

    np_img = image_data.numpy()

    explanation, segments_in_explanation, perturbation, new_class, confidence, orig_result, result = sedc_t2_mnist(
        device,
        np_img,
        model,
        segments,
        target_label,
        cf_mode)

    # print(new_class, confidence)


    if explanation is not None:
        mask_true_single = np.isin(segments, segments_in_explanation)
        # print(segments_in_explanation)
        # print(segments.shape)
        # print(mask_true_single)
        # mask_true_full = []
        # for row in mask_true_single:
        #     mask_true_col = []
        #     for col in row:
        #         if col:
        #             mask_true_col.append([1, 1, 1])
        #         else:
        #             mask_true_col.append([0, 0, 0])
        #     mask_true_full.append(mask_true_col)

        # mask_true_full = np.array(mask_true_full)

        # print(mask_true_full.shape,mask_true_full)

        # return image_data, (pil_img.convert("RGB") * (mask_true_full == 0) + (mask_true_full != 0) * (0, 1, 0)), perturbation
        return image_data, mask_true_single, perturbation, orig_result, result
        # return image_data, perturbation-image_data.numpy(), perturbation

    return image_data, None, None, orig_result, result

        