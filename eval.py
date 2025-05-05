import torch
import torch.nn.functional as F
from torchgeo.datasets import XView2
from transformers import UperNetForSemanticSegmentation, ConvNextImageProcessorFast
from tqdm import tqdm
import evaluate


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

precision = evaluate.load("precision")
recall = evaluate.load("recall")
mean_iou = evaluate.load("mean_iou")
f1 = evaluate.load("f1")

image_processor = ConvNextImageProcessorFast(do_resize=True, size=224, return_tensors="pt")

predictions, references = [], []
for pair in tqdm(test_ds):
    image_pair, mask_pair = pair["image"], pair["mask"]
    for image in image_pair:
        processed_image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        predictions.append(model(processed_image.unsqueeze(0)))
    for mask in mask_pair:
        interpolated = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(224, 224), mode='nearest')
        references.append(torch.round(interpolated).squeeze().long())

precision.add_batch(references=references, predictions=predictions)
recall.add_batch(references=references, predictions=predictions)
mean_iou.add_batch(references=references, predictions=predictions)
f1.add_batch(references=references, predictions=predictions)

with open("eval.txt", "w") as f:
    print(precision.compute(), file=f)
    print(recall.compute(), file=f)
    print(mean_iou.compute(), file=f)
    print(f1.compute(), file=f)
