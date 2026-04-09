# Q2.7 Final Pipeline Showcase

import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = "/content/da6401_assignment_2"
os.chdir(PROJECT_ROOT)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load models
# -----------------------------
classifier = VGG11Classifier(num_classes=37).to(device)
localizer = VGG11Localizer().to(device)
segmenter = VGG11UNet(num_classes=3, in_channels=3, dropout_p=0.5).to(device)

cls_ckpt = torch.load("checkpoints/classifier.pth", map_location=device, weights_only=False)
loc_ckpt = torch.load("checkpoints/localizer.pth", map_location=device, weights_only=False)
seg_ckpt = torch.load("checkpoints/unet.pth", map_location=device, weights_only=False)

if isinstance(cls_ckpt, dict) and "state_dict" in cls_ckpt:
    cls_ckpt = cls_ckpt["state_dict"]
if isinstance(loc_ckpt, dict) and "state_dict" in loc_ckpt:
    loc_ckpt = loc_ckpt["state_dict"]
if isinstance(seg_ckpt, dict) and "state_dict" in seg_ckpt:
    seg_ckpt = seg_ckpt["state_dict"]

classifier.load_state_dict(cls_ckpt)
localizer.load_state_dict(loc_ckpt)
segmenter.load_state_dict(seg_ckpt)

classifier.eval()
localizer.eval()
segmenter.eval()

# -----------------------------
# preprocessing
# -----------------------------
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(path, image_size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_norm = (img_np - mean) / std
    x = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0)
    return img_np, x

def colorize_mask(mask):
    mask = mask.astype(np.uint8)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color_mask[mask == 0] = [255, 0, 0]   # pet
    color_mask[mask == 1] = [0, 255, 0]   # border
    color_mask[mask == 2] = [0, 0, 255]   # background

    return color_mask

# -----------------------------
# W&B
# -----------------------------
wandb.init(project="da6401-assignment2", name="q27_final_pipeline_showcase")

table = wandb.Table(columns=[
    "Original Image",
    "Bounding Box Output",
    "Segmentation Output",
    "Predicted Class Index"
])

image_files = ["external_images/dog1.jpg", "external_images/dog2.jpg", "external_images/cat1.jpg"]

for fname in image_files:
    img_np, x = preprocess_image(fname)
    x = x.to(device)

    with torch.no_grad():
        cls_logits = classifier(x)
        bbox = localizer(x)[0].cpu().numpy()
        seg_logits = segmenter(x)
        pred_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

    pred_class = int(torch.argmax(cls_logits, dim=1).item())

    # -------- bbox image --------
    fig1, ax1 = plt.subplots(1, figsize=(5, 5))
    ax1.imshow(img_np)

    xc, yc, w, h = bbox
    rect = patches.Rectangle(
        (xc - w/2, yc - h/2),
        w,
        h,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax1.add_patch(rect)
    ax1.set_title(f"Predicted Class Index: {pred_class}")
    ax1.axis("off")

    # -------- segmentation image --------
    seg_color = colorize_mask(pred_mask)

    fig2, ax2 = plt.subplots(1, figsize=(5, 5))
    ax2.imshow(seg_color)
    ax2.set_title("Predicted Segmentation Mask")
    ax2.axis("off")

    table.add_data(
        wandb.Image(img_np, caption=fname),
        wandb.Image(fig1),
        wandb.Image(fig2),
        pred_class
    )

    plt.close(fig1)
    plt.close(fig2)

wandb.log({"Q2.7 Final Pipeline Showcase": table})
wandb.finish()

print("Q2.7 logged successfully.")
