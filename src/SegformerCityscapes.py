from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

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