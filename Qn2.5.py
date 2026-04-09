# Q2.5 Object Detection: Confidence & IoU
# Confidence is computed from classifier softmax on the predicted crop

import os
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = "/content/da6401_assignment_2"
os.chdir(PROJECT_ROOT)

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from models.classification import VGG11Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 1. Dataset
# If your local test split is incomplete, use trainval
# -------------------------------------------------
dataset = OxfordIIITPetDataset(
    root="data/oxford_pet",
    split="trainval",
    image_size=224
)

print("Dataset size:", len(dataset))

# -------------------------------------------------
# 2. Load localization model
# -------------------------------------------------
localizer = VGG11Localizer().to(device)
loc_ckpt = torch.load("checkpoints/localizer.pth", map_location=device, weights_only=False)
if isinstance(loc_ckpt, dict) and "state_dict" in loc_ckpt:
    loc_ckpt = loc_ckpt["state_dict"]
localizer.load_state_dict(loc_ckpt)
localizer.eval()

# -------------------------------------------------
# 3. Load classifier model
# -------------------------------------------------
classifier = VGG11Classifier(num_classes=37).to(device)
cls_ckpt = torch.load("checkpoints/classifier.pth", map_location=device, weights_only=False)
if isinstance(cls_ckpt, dict) and "state_dict" in cls_ckpt:
    cls_ckpt = cls_ckpt["state_dict"]
classifier.load_state_dict(cls_ckpt)
classifier.eval()

# -------------------------------------------------
# 4. Helpers
# -------------------------------------------------
def cxcywh_to_xyxy(box):
    xc, yc, w, h = box
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2

def compute_iou(box1, box2):
    x11, y11, x12, y12 = cxcywh_to_xyxy(box1)
    x21, y21, x22, y22 = cxcywh_to_xyxy(box2)

    xa = max(x11, x21)
    ya = max(y11, y21)
    xb = min(x12, x22)
    yb = min(y12, y22)

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)

    union = area1 + area2 - inter_area + 1e-8
    return inter_area / union

def clamp_box_xyxy(x1, y1, x2, y2, H=224, W=224):
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(1, min(W, int(round(x2))))
    y2 = max(1, min(H, int(round(y2))))

    # ensure non-empty crop
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)

    return x1, y1, x2, y2

def compute_detection_confidence_from_crop(image_tensor, pred_box):
    """
    image_tensor: [3,224,224] normalized tensor from dataset
    pred_box: [xc,yc,w,h] in pixel space
    Returns max softmax probability from classifier on predicted crop
    """
    x1, y1, x2, y2 = cxcywh_to_xyxy(pred_box)
    x1, y1, x2, y2 = clamp_box_xyxy(x1, y1, x2, y2, H=image_tensor.shape[1], W=image_tensor.shape[2])

    crop = image_tensor[:, y1:y2, x1:x2].unsqueeze(0).to(device)  # [1,3,h,w]
    crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = classifier(crop)
        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1).values.item()

    return float(conf)

# -------------------------------------------------
# 5. First pass: find mixed examples
# -------------------------------------------------
results = []

num_scan = min(120, len(dataset))
for i in range(num_scan):
    image, label, bbox, _ = dataset[i]
    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_box = localizer(x)[0].detach().cpu().numpy()

    gt_box = bbox.numpy()
    iou = compute_iou(pred_box, gt_box)
    confidence = compute_detection_confidence_from_crop(image, pred_box)

    results.append({
        "idx": i,
        "image": image,
        "gt_box": gt_box,
        "pred_box": pred_box,
        "iou": float(iou),
        "confidence": float(confidence),
    })

# sort by IoU so we include weak and strong examples
results_sorted = sorted(results, key=lambda r: r["iou"])
selected = results_sorted[:5] + results_sorted[-5:]
selected = sorted(selected, key=lambda r: r["iou"])

# -------------------------------------------------
# 6. W&B table
# -------------------------------------------------
wandb.init(project="da6401-assignment2", name="q25_detection_table_confidence")

table = wandb.Table(columns=[
    "Image",
    "Confidence",
    "IoU",
    "Comment"
])

for item in selected:
    image = item["image"]
    gt_box = item["gt_box"]
    pred_box = item["pred_box"]
    iou = item["iou"]
    confidence = item["confidence"]

    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(img)

    # GT box in green
    gx, gy, gw, gh = gt_box
    gt_rect = patches.Rectangle(
        (gx - gw / 2.0, gy - gh / 2.0),
        gw,
        gh,
        linewidth=2,
        edgecolor="green",
        facecolor="none"
    )
    ax.add_patch(gt_rect)

    # Predicted box in red
    px, py, pw, ph = pred_box
    pred_rect = patches.Rectangle(
        (px - pw / 2.0, py - ph / 2.0),
        pw,
        ph,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(pred_rect)

    if iou < 0.30:
        comment = "Failure case"
    elif iou < 0.50:
        comment = "Weak localization"
    elif iou < 0.75:
        comment = "Reasonable localization"
    else:
        comment = "Strong localization"

    ax.set_title(f"IoU={iou:.3f} | Conf={confidence:.3f}")
    ax.axis("off")

    table.add_data(
        wandb.Image(fig),
        float(confidence),
        float(iou),
        comment
    )

    plt.close(fig)

wandb.log({"Q2.5 Detection Table": table})
wandb.finish()

# Print likely failure case for report discussion
failure = selected[0]
print("Lowest-IoU example:")
print("Index:", failure["idx"])
print("IoU:", failure["iou"])
print("Confidence:", failure["confidence"])
