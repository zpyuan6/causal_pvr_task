import matplotlib.pyplot as plt
import numpy as np

def present_heatmap(input_img:np.ndarray, heatmap:np.ndarray, predict=None, class_idx=None):
    if input_img.shape[-1]!=3:
        input_img = input_img.transpose(1,2,0)

    plt.subplot(1,2,1)
    if predict!=None:
        print(predict, class_idx)
        plt.ylabel(f"Output: {class_idx} {round(predict[0][class_idx].item()*100,2)}% ")
    plt.imshow(input_img)
    ax =plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    # plt.subplot(1,3,2)
    # plt.imshow(heatmap)
    # plt.axis('off')
    plt.subplot(1,2,2)
    # plt.imshow(weighted_result)
    plt.imshow(input_img)
    plt.imshow(heatmap, alpha=0.5)
    plt.axis('off')
    plt.show()



