"""Segmentation model"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class DoubleConv(nn.Module):
    """Two convolution blocks with BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample with transposed convolution, concatenate skip, then double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.dropout = CustomDropout(dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.up1 = UpBlock(512, 512, 512, dropout_p=dropout_p)
        self.up2 = UpBlock(512, 512, 256, dropout_p=dropout_p)
        self.up3 = UpBlock(256, 256, 128, dropout_p=dropout_p)
        self.up4 = UpBlock(128, 128, 64, dropout_p=dropout_p)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64 + 64, num_classes, kernel_size=1)

    def set_transfer_strategy(self, strategy: str = "full"):
        """
        Set transfer learning strategy for encoder.

        Args:
            strategy:
                - "strict": freeze entire encoder
                - "partial": freeze early encoder blocks only
                - "full": train entire model
        """
        # first make everything trainable
        for p in self.parameters():
            p.requires_grad = True

        if strategy == "strict":
            for p in self.encoder.parameters():
                p.requires_grad = False

        elif strategy == "partial":
            # freeze early blocks only
            # assumes VGG11Encoder has block1, block2, block3, block4, block5
            for block_name in ["block1", "block2", "block3"]:
                block = getattr(self.encoder, block_name, None)
                if block is not None:
                    for p in block.parameters():
                        p.requires_grad = False

        elif strategy == "full":
            pass

        else:
            raise ValueError(f"Unknown transfer strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.encoder(x, return_features=True)

        x1 = features["x1"]   # [B, 64, 224, 224]
        x2 = features["x2"]   # [B, 128, 112, 112]
        x3 = features["x3"]   # [B, 256, 56, 56]
        x4 = features["x4"]   # [B, 512, 28, 28]
        x5 = features["x5"]   # [B, 512, 14, 14]

        d1 = self.up1(bottleneck, x5)   # [B, 512, 14, 14]
        d2 = self.up2(d1, x4)           # [B, 256, 28, 28]
        d3 = self.up3(d2, x3)           # [B, 128, 56, 56]
        d4 = self.up4(d3, x2)           # [B, 64, 112, 112]
        d5 = self.up5(d4)               # [B, 64, 224, 224]

        out = torch.cat([d5, x1], dim=1)  # [B, 128, 224, 224]
        out = self.final_conv(out)        # [B, num_classes, 224, 224]

        return out
