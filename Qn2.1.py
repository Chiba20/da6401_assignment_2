import os
import torch
import matplotlib.pyplot as plt
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier

# ---------------------------
# CONFIG
# ---------------------------
DATA_ROOT = "data/oxford_pet"
MODEL_PATH = "checkpoints/classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD DATA (1 image only)
# ---------------------------
dataset = OxfordIIITPetDataset(
    root=DATA_ROOT,
    split="trainval",
    image_size=224
)

img, _, _, _ = dataset[0]
img = img.unsqueeze(0).to(DEVICE)

# ---------------------------
# BUILD MODELS
# ---------------------------
model_bn = VGG11Classifier(num_classes=37, dropout_p=0.5).to(DEVICE)
model_no_bn = VGG11Classifier(num_classes=37, dropout_p=0.5).to(DEVICE)

# ---------------------------
# REMOVE BATCHNORM FROM ONE MODEL
# ---------------------------
def remove_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, torch.nn.Identity())
        else:
            remove_batchnorm(child)

remove_batchnorm(model_no_bn)

# ---------------------------
# LOAD TRAINED WEIGHTS
# ---------------------------
model_bn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_no_bn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model_bn.eval()
model_no_bn.eval()

# ---------------------------
# HOOK ACTIVATIONS
# ---------------------------
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Adjust index if needed
model_bn.features[4].register_forward_hook(get_activation("bn"))
model_no_bn.features[4].register_forward_hook(get_activation("no_bn"))

# ---------------------------
# FORWARD PASS
# ---------------------------
_ = model_bn(img)
act_bn = activations["bn"].cpu().numpy().flatten()

_ = model_no_bn(img)
act_no_bn = activations["no_bn"].cpu().numpy().flatten()

# ---------------------------
# PLOT
# ---------------------------
plt.figure(figsize=(8,5))

plt.hist(act_no_bn, bins=50, alpha=0.5, label="Without BatchNorm")
plt.hist(act_bn, bins=50, alpha=0.5, label="With BatchNorm")

plt.legend()
plt.title("Activation Distribution (3rd Conv Layer)")
plt.xlabel("Activation Values")
plt.ylabel("Frequency")

plt.savefig("activation_distribution.png")
plt.show()

# ---------------------------
# LOG TO WANDB
# ---------------------------
wandb.init(project="da6401-assignment2", name="bn-analysis")

wandb.log({
    "Activation Distribution": wandb.Image("activation_distribution.png")
})

wandb.finish()

print("Done! Plot saved and uploaded.")
