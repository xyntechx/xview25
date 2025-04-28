from tqdm import tqdm
import numpy as np
import torch
from torchgeo.datasets import XView2
from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
import evaluate
from SegmentationDataset import SegmentationDataset


def load_model():
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
    return model


def load_image_processor():
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
    return image_processor


def load_data():
    DATA_PATH = "../../../datasets/xview2/current"
    train = XView2(DATA_PATH, split="train")
    test = XView2(DATA_PATH, split="test")

    train_images, train_masks, test_images, test_masks = [], [], [], []

    for d in tqdm(train):
        # Pre- and post-disaster images/masks are coupled
        train_images.append(d["image"][0])
        train_masks.append(d["mask"][0])

        train_images.append(d["image"][1])
        train_masks.append(d["mask"][1])

    for d in tqdm(test):
        # Pre- and post-disaster images/masks are coupled
        test_images.append(d["image"][0])
        test_masks.append(d["mask"][0])

        test_images.append(d["image"][1])
        test_masks.append(d["mask"][1])

    return train_images, train_masks, test_images, test_masks


def load_trainer():
    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=5e-4,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=1,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        fp16=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    return trainer


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric_mean_iou.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=5,
            ignore_index=255,
            reduce_labels=False,
        )
        metrics.update(metric_precision.compute(predictions=pred_labels, references=labels, average=None))
        metrics.update(metric_recall.compute(predictions=pred_labels, references=labels, average=None))
        metrics.update(metric_f1.compute(predictions=pred_labels, references=labels, average=None))

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()

        return metrics


if __name__ == "__main__":
    train_images, train_masks, test_images, test_masks = load_data()
    image_processor = load_image_processor()
    model = load_model()

    train_ds = SegmentationDataset(train_images, train_masks, image_processor)
    test_ds = SegmentationDataset(test_images, test_masks, image_processor)

    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_mean_iou = evaluate.load("mean_iou")
    metric_f1 = evaluate.load("f1")

    trainer = load_trainer()

    trainer.train()
