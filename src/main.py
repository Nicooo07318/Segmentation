from PIL import Image
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# restructure classes and pair the new class id with a readable name
ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
            'train', 'motorcycle', 'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
print(n_classes)
print(class_map)


# give each class id a specific color
colors = [[128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(n_classes), colors))

# remove unwanted classes and  
def encode_segmap(mask):
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

# turns map of class indices into an RGB image
def decode_segmap(temp):
    #convert gray scale to color
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)
    for l in range(n_classes):
        rgb[temp==l] = np.array(label_colours[l]) / 255.0
    return rgb


# set device and dataset
def init():
    device = "mps" if torch.mps.is_available() else "cuda"
    dataset = Cityscapes('Data/CityScapes/', split='val', mode='fine', target_type='semantic')

    name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name)
    model.eval()
    model.to(device)


    return device, dataset, processor, model




def main():
    device, dataset, processor, model = init()
    eval(dataset, device, processor, model)



def eval(dataset, device, processor, model):
    img, mask = dataset[0]

    # visualize img and mask
    # fig,ax=plt.subplots(ncols=2,figsize=(12,8))
    # ax[0].imshow(dataset[0][0])
    # ax[1].imshow(dataset[0][1],cmap='gray')
    # plt.show()


    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits back to the original 2048x1024 and take the argmax
    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]   # (height, width)
    )[0].cpu().numpy()   # shape (1024, 2048), values 0-18


    gt = encode_segmap(np.array(mask))

    #pred_col = decode_segmap(pred)

    fig,ax=plt.subplots(ncols=3,figsize=(12,10))
    ax[0].imshow(img); ax[0].set_title("Image")
    ax[1].imshow(decode_segmap(pred),cmap='gray'); ax[1].set_title("Prediction")
    ax[2].imshow(decode_segmap(gt)); ax[2].set_title("Ground truth")
    plt.show()






    



if __name__ == '__main__':
    main()