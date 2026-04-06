"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Unified multi-task model.

    Loads separately trained models for:
    - classification
    - localization
    - segmentation

    Checkpoints are downloaded from Google Drive at runtime using gdown.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        # Keep checkpoints inside checkpoints folder
        checkpoints_dir = "checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)

        classifier_path = os.path.join(checkpoints_dir, os.path.basename(classifier_path))
        localizer_path = os.path.join(checkpoints_dir, os.path.basename(localizer_path))
        unet_path = os.path.join(checkpoints_dir, os.path.basename(unet_path))

        # Download checkpoints only if missing
        import gdown

        if not os.path.exists(classifier_path):
            gdown.download(
                url="https://drive.google.com/file/d/1CfJ_6l3gL3lBOQqmEHPs0DEh2lnuPPa1/view?usp=sharing",
                output=classifier_path,
                quiet=False,
                fuzzy=True,
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                url="https://drive.google.com/file/d/1x__FeeXEyU7eC3l7et9WvmvekGAeDPWC/view?usp=sharing",
                output=localizer_path,
                quiet=False,
                fuzzy=True,
            )

        if not os.path.exists(unet_path):
            gdown.download(
                url="https://drive.google.com/file/d/1M-qy4231U_XXUbBpQvfKL2Y1FFvtL0eP/view?usp=sharing",
                output=unet_path,
                quiet=False,
                fuzzy=True,
            )

        # Build full task-specific models
        self.classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels
        )
        self.localizer_model = VGG11Localizer(
            in_channels=in_channels
        )
        self.segmenter_model = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels
        )

        # Load classifier checkpoint
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        if isinstance(classifier_ckpt, dict) and "state_dict" in classifier_ckpt:
            classifier_ckpt = classifier_ckpt["state_dict"]
        self.classifier_model.load_state_dict(classifier_ckpt)

        # Load localizer checkpoint
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        if isinstance(localizer_ckpt, dict) and "state_dict" in localizer_ckpt:
            localizer_ckpt = localizer_ckpt["state_dict"]
        self.localizer_model.load_state_dict(localizer_ckpt)

        # Load segmentation checkpoint
        unet_ckpt = torch.load(unet_path, map_location="cpu")
        if isinstance(unet_ckpt, dict) and "state_dict" in unet_ckpt:
            unet_ckpt = unet_ckpt["state_dict"]
        self.segmenter_model.load_state_dict(unet_ckpt)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Dictionary with:
            - 'classification': logits [B, num_breeds]
            - 'localization': boxes [B, 4]
            - 'segmentation': logits [B, seg_classes, H, W]
        """
        classification_logits = self.classifier_model(x)
        localization_output = self.localizer_model(x)
        segmentation_logits = self.segmenter_model(x)

        return {
            "classification": classification_logits,
            "localization": localization_output,
            "segmentation": segmentation_logits,
        }
