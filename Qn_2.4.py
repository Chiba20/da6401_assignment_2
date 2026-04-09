# Q2.4 - W&B logging for feature maps

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

# -------------------------------------------------
# 1. Go to project root
# -------------------------------------------------
PROJECT_ROOT = "/content/da6401_assignment_2"
os.chdir(PROJECT_ROOT)

# -------------------------------------------------
# 2. Imports from your project
# -------------------------------------------------
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 3. Load dataset and pick ONE DOG image
#    dogs -> lowercase names
# -------------------------------------------------
dataset = OxfordIIITPetDataset(
    root="data/oxford_pet",
    split="trainval",
    image_size=224
)

dog_idx = None
dog_name = None

for i, (image_name, class_id) in enumerate(dataset.samples):
    if image_name[0].islower():
        dog_idx = i
        dog_name = image_name
        break

if dog_idx is None:
    raise ValueError("No dog image found in dataset.")

print("Selected DOG image:", dog_name, "| index:", dog_idx)

image, label, _, _ = dataset[dog_idx]
x = image.unsqueeze(0).to(device)

# -------------------------------------------------
# 4. Load trained classifier checkpoint
# -------------------------------------------------
ckpt_path = "checkpoints/classifier.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError("classifier.pth not found inside checkpoints/")

model = VGG11Classifier(num_classes=37).to(device)
ckpt = torch.load(ckpt_path, map_location=device)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

model.load_state_dict(ckpt)
model.eval()

# -------------------------------------------------
# 5. Capture feature maps
# -------------------------------------------------
conv_layers = [m for m in model.encoder.modules() if isinstance(m, nn.Conv2d)]
first_conv = conv_layers[0]
last_conv = conv_layers[-1]

activations = {}

def save_activation(name):
    def hook(module, inp, out):
        activations[name] = out.detach().cpu()
    return hook

h1 = first_conv.register_forward_hook(save_activation("first"))
h2 = last_conv.register_forward_hook(save_activation("last"))

with torch.no_grad():
    _ = model(x)

h1.remove()
h2.remove()

# -------------------------------------------------
# 6. Prepare input image for display
# -------------------------------------------------
img = image.permute(1, 2, 0).cpu().numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# -------------------------------------------------
# 7. Function to create ONE clean figure for feature maps
# -------------------------------------------------
def make_feature_map_figure(feat, title, max_maps=8):
    feat = feat[0]   # remove batch dim
    n = min(max_maps, feat.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(8):
        axes[i].axis("off")

    for i in range(n):
        fmap = feat[i].numpy()
        axes[i].imshow(fmap, cmap="gray")
        axes[i].set_title(f"Map {i+1}", fontsize=9)
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

# -------------------------------------------------
# 8. Create figures
# -------------------------------------------------
fig_input = plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title(f"Input Dog Image: {dog_name}")
plt.tight_layout()

fig_first = make_feature_map_figure(
    activations["first"],
    "Feature Maps from First Convolution Layer",
    max_maps=8
)

fig_last = make_feature_map_figure(
    activations["last"],
    "Feature Maps from Last Convolution Layer (Before Pooling)",
    max_maps=8
)

# -------------------------------------------------
# 9. Log correctly to W&B
# -------------------------------------------------
wandb.init(project="da6401-assignment2", name="q24_feature_maps_correct")

wandb.log({
    "Q2.4 Input Dog Image": wandb.Image(fig_input),
    "Q2.4 First Conv Feature Maps": wandb.Image(fig_first),
    "Q2.4 Last Conv Feature Maps": wandb.Image(fig_last),
})

wandb.finish()

# optional: show in notebook too
plt.show()
