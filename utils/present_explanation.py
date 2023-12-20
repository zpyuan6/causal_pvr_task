import matplotlib.pyplot as plt
import numpy as np

def present_heatmap(input_img:np.ndarray, heatmap:np.ndarray):
    if input_img.shape[-1]!=3:
        input_img = input_img.transpose(1,2,0)

    plt.subplot(1,3,1)
    plt.imshow(input_img)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(heatmap)
    plt.axis('off')
    plt.subplot(1,3,3)
    # plt.imshow(weighted_result)
    plt.imshow(input_img)
    plt.imshow(heatmap, alpha=0.5)
    plt.axis('off')
    plt.show()



