import torch
import torch.nn.functional as F
from torchgeo.datasets import XView2
from transformers import ConvNextImageProcessorFast, UperNetConfig, UperNetForSemanticSegmentation, TrainingArguments, Trainer
import argparse


def main(epochs=10, batch_size=16, backbone="facebook/convnextv2-tiny-1k-224"):
    def collate_fn(examples):
        images, masks = [], []
        for pair in examples:
            image_pair, mask_pair = pair["image"], pair["mask"]
            for image in image_pair:
                processed_image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                images.append(processed_image)
            for mask in mask_pair:
                interpolated = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(224, 224), mode='nearest')
                masks.append(torch.round(interpolated).squeeze().long())
        pixel_values = torch.stack(images)
        labels = torch.stack(masks)
        return {"pixel_values": pixel_values, "labels": labels}

    run_id = f"{backbone.split()[1]}_e{epochs}_bs{batch_size}"

    image_processor = ConvNextImageProcessorFast(do_resize=True, size=224, return_tensors="pt")
    config = UperNetConfig(
        backbone=backbone,
        use_pretrained_backbone=True,
        backbone_kwargs={
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
        },
        num_labels=5,
        id2label={
            0: "background",
            1: "no damage",
            2: "minor damage",
            3: "major damage",
            4: "destroyed"
        },
        label2id = {
            "background": 0,
            "no damage": 1,
            "minor damage": 2,
            "major damage": 3,
            "destroyed": 4
        }
    )
    model = UperNetForSemanticSegmentation(config)

    training_args = TrainingArguments(
        run_name=f"xview25_{run_id}",
        output_dir=f"output_{run_id}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=1,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )

    DATA_PATH = "../../../datasets/xview2/current"
    train_ds = XView2(DATA_PATH, split="train")
    test_ds = XView2(DATA_PATH, split="test")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(f"model_{run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default=10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default=16)")
    parser.add_argument("--backbone", type=str, default="facebook/convnextv2-tiny-1k-224", help="Backbone model HF path (default='facebook/convnextv2-tiny-1k-224')")

    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size, backbone=args.backbone)
