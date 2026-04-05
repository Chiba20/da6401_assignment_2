"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        # Keep checkpoints inside the checkpoints folder
        checkpoints_dir = "checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)

        classifier_path = os.path.join(checkpoints_dir, os.path.basename(classifier_path))
        localizer_path = os.path.join(checkpoints_dir, os.path.basename(localizer_path))
        unet_path = os.path.join(checkpoints_dir, os.path.basename(unet_path))

        # Download checkpoints from Google Drive only if they are missing
        import gdown

        if not os.path.exists(classifier_path):
            gdown.download(
                url="https://drive.google.com/uc?id=1uklRkhOAQy6R0qOAgCMdfmmdvYm5LwB8",
                output=classifier_path,
                quiet=False,
                fuzzy=True,
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                url="https://drive.google.com/uc?id=17aI4hlRVe26vIdaOKfl8RxtNdxpMZTxZ",
                output=localizer_path,
                quiet=False,
                fuzzy=True,
            )

        if not os.path.exists(unet_path):
            gdown.download(
                url="https://drive.google.com/uc?id=1M-qy4231U_XXUbBpQvfKL2Y1FFvtL0eP",
                output=unet_path,
                quiet=False,
                fuzzy=True,
            )

        # Build individual models
        classifier_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer_model = VGG11Localizer(in_channels=in_channels)
        segmentation_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load classifier checkpoint
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        if isinstance(classifier_ckpt, dict) and "state_dict" in classifier_ckpt:
            classifier_ckpt = classifier_ckpt["state_dict"]
        classifier_model.load_state_dict(classifier_ckpt)

        # Load localizer checkpoint
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        if isinstance(localizer_ckpt, dict) and "state_dict" in localizer_ckpt:
            localizer_ckpt = localizer_ckpt["state_dict"]
        localizer_model.load_state_dict(localizer_ckpt)

        # Load segmentation checkpoint
        unet_ckpt = torch.load(unet_path, map_location="cpu")
        if isinstance(unet_ckpt, dict) and "state_dict" in unet_ckpt:
            unet_ckpt = unet_ckpt["state_dict"]
        segmentation_model.load_state_dict(unet_ckpt)

        # Shared backbone from classifier
        self.encoder = classifier_model.encoder

        # Classification head from trained classifier
        self.classifier = classifier_model.classifier

        # Localization head from trained localizer
        self.localizer = localizer_model.localizer

        # Full trained segmentation model
        self.segmenter = segmentation_model

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        features = self.encoder(x)

        classification_logits = self.classifier(features)
        localization_output = self.localizer(features)
        segmentation_logits = self.segmenter(x)

        return {
            "classification": classification_logits,
            "localization": localization_output,
            "segmentation": segmentation_logits,
        }
