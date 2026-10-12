from PIL import Image
from torchvision.datasets import Cityscapes
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import evaluate


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import TrainingArguments, Trainer
from torch.utils.data import random_split

class SegformerCityscapes(Dataset):
    def __init__(self, root, split, processor):
        self.dataset = Cityscapes(
            root, 
            split=split,
            mode="fine",
            target_type="semantic"
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        mask = encode_segmap(np.array(mask))

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
        }



device = "mps" if torch.mps.is_available() else "cuda"

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


name = "nvidia/mit-b0"
processor = SegformerImageProcessor.from_pretrained(
    name,
    reduce_labels=False,
)

id2label = dict(enumerate(class_names))
label2id = {name: index for index, name in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    name,
    num_labels=n_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,)
model.to(device)


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
    
    train_dataset = SegformerCityscapes('' \
        'Data/CityScapes/', 
        split='train', 
        processor=processor)
    
    full_val_dataset = SegformerCityscapes(
        'Data/CityScapes/', 
        split='val', 
        processor=processor)

    val_size = int(0.8*len(full_val_dataset))
    test_size = len(full_val_dataset) - val_size

    val_dataset, test_dataset = random_split(
        full_val_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )


    return val_dataset, train_dataset, test_dataset

def train_setup():
    epochs = 50
    lr = 0.00006
    batch_size = 2

    hub_model_id = "segformer-b0-finetuned-Cityscapes-50epochs"

    training_args = TrainingArguments(
        "segformer-b0-finetuned-Cityscapes-50epochs-output",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        push_to_hub=False,
        hub_model_id=hub_model_id,
        hub_strategy="end",
    )
    return training_args




def main():
    val_dataset, train_dataset, test_dataset = init()
    
    training_args = train_setup()
    train(training_args, train_dataset, val_dataset, test_dataset)

    
    #eval(val_dataset, processor, model)



def eval(dataset, processor, model):
    model.eval()
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


def train(training_args, train_dataset, val_dataset, test_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()



metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=ignore_index,
        reduce_labels=processor.do_reduce_labels,
    )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics


    



if __name__ == '__main__':
    main()