from PIL import Image
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation





def main():
    print()
    device = "mps" if torch.mps.is_available() else "cuda"
    dataset = Cityscapes('Data/CityScapes/', split='test', mode='fine', target_type='semantic')
    #print(dataset[0][0].size)

    eval(dataset, device)



def eval(dataset, device):
    img = dataset[1][0]
    mask = dataset[1][1]

    name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name)
    model.eval()
    model.to(device)

    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits back to the original 2048x1024 and take the argmax
    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]   # (height, width)
    )[0].cpu().numpy()   # shape (1024, 2048), values 0-18


    palette = np.array([
        [128, 64,128],[244, 35,232],[ 70, 70, 70],[102,102,156],[190,153,153],
        [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],
        [ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],
        [  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]
    ], dtype=np.uint8)

    color_mask = palette[pred]   # (H, W, 3)



    overlay = (0.55 * np.array(img) + 0.45 * color_mask).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for a, im, t in zip(ax, [img, color_mask, overlay],
                        ["Input", "Prediction", "Overlay"]):
        a.imshow(im); a.set_title(t); a.axis("off")
    plt.tight_layout()
    #plt.savefig("segformer_result.png", dpi=200)
    #plt.show()

    labels = [model.config.id2label[i] for i in range(19)]
    handles = [mpatches.Patch(color=palette[i]/255, label=labels[i]) for i in range(19)]
    plt.figure(figsize=(3, 6)); plt.legend(handles=handles, loc="center"); plt.axis("off")

    plt.show()


if __name__ == '__main__':
    main()