import torch
import torch.nn.functional as F
from torchgeo.datasets import XView2
from torchvision.transforms.functional import convert_image_dtype
from transformers import UperNetForSemanticSegmentation, TrainingArguments, Trainer
import evaluate


def collate_fn(examples):
    images, masks = [], []
    for pair in examples:
        image_pair, mask_pair = pair["image"], pair["mask"]
        for image in image_pair:
            images.append(convert_image_dtype(image))
        for mask in mask_pair:
            masks.append(mask)

    # images = torch.stack([example["image"].to(memory_format=torch.channels_last) for example in examples])
    # masks = torch.stack([example["mask"] for example in examples])
    return {"image": torch.stack(images), "mask": torch.stack(masks)}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(inputs["image"])
        target = inputs["mask"]
        loss = F.binary_cross_entropy(F.sigmoid(outputs), target)
        return (loss, outputs) if return_outputs else loss


model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
model.config.id2label = {
    0: "background",
    1: "no damage",
    2: "minor damage",
    3: "major damage",
    4: "destroyed"
}
model.config.label2id = {
    "background": 0,
    "no damage": 1,
    "minor damage": 2,
    "major damage": 3,
    "destroyed": 4
}

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    eval_strategy="epoch",
    remove_unused_columns=False,
    dataloader_num_workers=1,
)

DATA_PATH = "../../../datasets/xview2/current"
train_ds = XView2(DATA_PATH, split="train")
test_ds = XView2(DATA_PATH, split="test")

trainer = MyTrainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collate_fn,
)

trainer.train()
