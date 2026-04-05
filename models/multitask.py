"""Unified multi-task model"""

import os
import torch
import torch.nn as nn
import gdown

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import UpBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.5,
    ):
        """
        Initialize the shared backbone and task-specific heads.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained UNet weights.
            dropout_p: Dropout probability.
        """
        super().__init__()

        # make sure checkpoints folder exists
        os.makedirs("checkpoints", exist_ok=True)

        # download required checkpoints from Google Drive
        if not os.path.exists(classifier_path):
            gdown.download(
                "https://drive.google.com/uc?id=1uklRkhOAQy6R0qOAgCMdfmmdvYm5LwB8",
                classifier_path,
                quiet=False,
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                "https://drive.google.com/uc?id=17aI4hlRVe26vIdaOKfl8RxtNdxpMZTxZ",
                localizer_path,
                quiet=False,
            )

        if not os.path.exists(unet_path):
            gdown.download(
                "https://drive.google.com/uc?id=1M-qy4231U_XXUbBpQvfKL2Y1FFvtL0eP",
                unet_path,
                quiet=False,
            )

        # shared encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # classification head
        self.cls_avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # localization head
        self.loc_avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

        # segmentation head
        self.up1 = UpBlock(512, 512, 512, dropout_p=dropout_p)
        self.up2 = UpBlock(512, 512, 256, dropout_p=dropout_p)
        self.up3 = UpBlock(256, 256, 128, dropout_p=dropout_p)
        self.up4 = UpBlock(128, 128, 64, dropout_p=dropout_p)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_seg_conv = nn.Conv2d(64 + 64, seg_classes, kernel_size=1)

        # load trained weights from single-task models
        classifier_state = torch.load(classifier_path, map_location="cpu")
        localizer_state = torch.load(localizer_path, map_location="cpu")
        unet_state = torch.load(unet_path, map_location="cpu")

        # load encoder weights from classifier
        self.encoder.load_state_dict(
            {
                k.replace("encoder.", ""): v
                for k, v in classifier_state.items()
                if k.startswith("encoder.")
            },
            strict=True,
        )

        # load classifier head
        self.classifier_head.load_state_dict(
            {
                k.replace("classifier.", ""): v
                for k, v in classifier_state.items()
                if k.startswith("classifier.")
            },
            strict=True,
        )

        # load localization head
        self.localization_head.load_state_dict(
            {
                k.replace("regressor.", ""): v
                for k, v in localizer_state.items()
                if k.startswith("regressor.")
            },
            strict=True,
        )

        # load segmentation decoder/head
        self.up1.load_state_dict(
            {
                k.replace("up1.", ""): v
                for k, v in unet_state.items()
                if k.startswith("up1.")
            },
            strict=True,
        )
        self.up2.load_state_dict(
            {
                k.replace("up2.", ""): v
                for k, v in unet_state.items()
                if k.startswith("up2.")
            },
            strict=True,
        )
        self.up3.load_state_dict(
            {
                k.replace("up3.", ""): v
                for k, v in unet_state.items()
                if k.startswith("up3.")
            },
            strict=True,
        )
        self.up4.load_state_dict(
            {
                k.replace("up4.", ""): v
                for k, v in unet_state.items()
                if k.startswith("up4.")
            },
            strict=True,
        )
        self.up5.load_state_dict(
            {
                k.replace("up5.", ""): v
                for k, v in unet_state.items()
                if k.startswith("up5.")
            },
            strict=True,
        )
        self.final_seg_conv.load_state_dict(
            {
                k.replace("final_conv.", "final_seg_conv."): v
                for k, v in unet_state.items()
                if k.startswith("final_conv.")
            },
            strict=False,
        )

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        bottleneck, features = self.encoder(x, return_features=True)

        # classification branch
        cls_feat = self.cls_avgpool(bottleneck)
        cls_logits = self.classifier_head(cls_feat)

        # localization branch
        loc_feat = self.loc_avgpool(bottleneck)
        bbox = self.localization_head(loc_feat)

        # segmentation branch
        x1 = features["x1"]
        x2 = features["x2"]
        x3 = features["x3"]
        x4 = features["x4"]
        x5 = features["x5"]

        d1 = self.up1(bottleneck, x5)
        d2 = self.up2(d1, x4)
        d3 = self.up3(d2, x3)
        d4 = self.up4(d3, x2)
        d5 = self.up5(d4)
        seg_logits = self.final_seg_conv(torch.cat([d5, x1], dim=1))

        return {
            "classification": cls_logits,
            "localization": bbox,
            "segmentation": seg_logits,
        }