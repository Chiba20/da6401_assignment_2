"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout."""

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError("Dropout probability p must be in [0, 1].")

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying inverted dropout during training.
        """
        if not self.training or self.p == 0.0:
            return x

        if self.p == 1.0:
            return torch.zeros_like(x)

        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).float()
        return (x * mask) / keep_prob