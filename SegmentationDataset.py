from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, image_processor):
        self.images = images
        self.masks = masks
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        processed = self.image_processor(
            image,
            mask,
            return_tensors="pt",
            input_data_format="channels_first"
        )

        return {
            "pixel_values": processed["pixel_values"][0],
            "labels": processed["labels"][0]
        }
