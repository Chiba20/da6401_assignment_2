# Q2.6 Segmentation Evaluation: Dice vs Pixel Accuracy

import os
import torch
import wandb
import numpy as np

PROJECT_ROOT = "/content/da6401_assignment_2"
os.chdir(PROJECT_ROOT)

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Dataset
# -----------------------------
dataset = OxfordIIITPetDataset(
    root="data/oxford_pet",
    split="trainval",
    image_size=224
)

# -----------------------------
# Load trained segmentation model
# -----------------------------
model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=0.5).to(device)
ckpt = torch.load("checkpoints/unet.pth", map_location=device, weights_only=False)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

model.load_state_dict(ckpt)
model.eval()

# -----------------------------
# Metrics
# -----------------------------
def pixel_accuracy(pred_mask, true_mask):
    return (pred_mask == true_mask).float().mean().item()

def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    dices = []
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()

        inter = (pred_c * true_c).sum()
        union = pred_c.sum() + true_c.sum()

        dice = (2 * inter + eps) / (union + eps)
        dices.append(dice.item())

    return sum(dices) / len(dices)


def colorize_mask(mask):
    mask = mask.astype(np.uint8)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # pet = black
    color_mask[mask == 0] = [0, 0, 0]

    # border = white
    color_mask[mask == 1] = [128, 120, 128]

    # background = grey
    color_mask[mask == 2] = [255, 255, 255]

    return color_mask

# -----------------------------
# W&B
# -----------------------------
wandb.init(project="da6401-assignment2", name="q26_segmentation_eval_colored")

table = wandb.Table(columns=[
    "Original Image",
    "Ground Truth Trimap",
    "Predicted Trimap",
    "Pixel Accuracy",
    "Dice Score"
])

# -----------------------------
# Log 5 samples
# -----------------------------
for i in range(5):
    image, label, bbox, mask = dataset[i]

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu()

    true_mask = mask.cpu()

    pa = pixel_accuracy(pred_mask, true_mask)
    ds = dice_score(pred_mask, true_mask)

    # original image for display
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # colorized masks
    gt_color = colorize_mask(true_mask.numpy())
    pred_color = colorize_mask(pred_mask.numpy())

    table.add_data(
        wandb.Image(img, caption="Original"),
        wandb.Image(gt_color, caption="Ground Truth"),
        wandb.Image(pred_color, caption="Prediction"),
        float(pa),
        float(ds)
    )

wandb.log({"Q2.6 Segmentation Table": table})
wandb.finish()

print("Q2.6 logged successfully with colored masks.")
