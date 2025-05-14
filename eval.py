import torch
import torch.nn.functional as F
from torchgeo.datasets import XView2
from transformers import UperNetForSemanticSegmentation, ConvNextImageProcessorFast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

model = UperNetForSemanticSegmentation.from_pretrained("model")
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

DATA_PATH = "../../../datasets/xview2/current"
test_ds = XView2(DATA_PATH, split="test")

image_processor = ConvNextImageProcessorFast(do_resize=True, size=224, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

predictions = []
masks = []

with torch.no_grad():
    for pair in tqdm(test_ds):
        image_pair, mask_pair = pair["image"], pair["mask"]
        for i, image in enumerate(image_pair):
            processed_image = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            outputs = model(processed_image)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            mask = mask_pair[i]
            mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(224, 224), mode="nearest")
            mask = torch.round(mask).squeeze().long().numpy()

            predictions.append(pred)
            masks.append(mask)

predictions_flat = np.concatenate([pred.flatten() for pred in predictions])
masks_flat = np.concatenate([mask.flatten() for mask in masks])

unique_preds = np.unique(predictions_flat)
unique_masks = np.unique(masks_flat)
print(f"Unique prediction values: {unique_preds}")
print(f"Unique mask values: {unique_masks}")

print("Calculating metrics...")

NUM_CLASSES = 5

precision = precision_score(masks_flat, predictions_flat, average="macro", zero_division=0, labels=list(range(NUM_CLASSES)))
recall = recall_score(masks_flat, predictions_flat, average="macro", zero_division=0, labels=list(range(NUM_CLASSES)))
f1 = f1_score(masks_flat, predictions_flat, average="macro", zero_division=0, labels=list(range(NUM_CLASSES)))

iou_per_class = jaccard_score(masks_flat, predictions_flat, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
mean_iou = np.mean(iou_per_class)

class_precision = precision_score(masks_flat, predictions_flat, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
class_recall = recall_score(masks_flat, predictions_flat, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
class_f1 = f1_score(masks_flat, predictions_flat, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

with open("eval.txt", "w") as f:
    print(f"Mean IoU: {mean_iou}", file=f)
    print(f"F1 Score (macro): {f1}", file=f)
    print(f"Precision: {precision}", file=f)
    print(f"Recall: {recall}", file=f)
    print("\nMetrics per class:", file=f)
    for i in range(NUM_CLASSES):
        class_name = model.config.id2label[i]
        print(f"Class {i} ({class_name}):", file=f)
        print(f"  IoU: {iou_per_class[i]}", file=f)
        print(f"  Precision: {class_precision[i]}", file=f)
        print(f"  Recall: {class_recall[i]}", file=f)
        print(f"  F1: {class_f1[i]}", file=f)

print("Eval results saved to eval.txt")
