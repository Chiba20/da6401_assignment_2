"""Custom IoU loss"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output:
                       'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        # Clamp widths and heights to avoid invalid boxes
        pred_xc, pred_yc, pred_w, pred_h = pred_boxes.unbind(dim=1)
        tgt_xc, tgt_yc, tgt_w, tgt_h = target_boxes.unbind(dim=1)

        pred_w = torch.clamp(pred_w, min=self.eps)
        pred_h = torch.clamp(pred_h, min=self.eps)
        tgt_w = torch.clamp(tgt_w, min=self.eps)
        tgt_h = torch.clamp(tgt_h, min=self.eps)

        # Convert (xc, yc, w, h) -> (x1, y1, x2, y2)
        pred_x1 = pred_xc - pred_w / 2.0
        pred_y1 = pred_yc - pred_h / 2.0
        pred_x2 = pred_xc + pred_w / 2.0
        pred_y2 = pred_yc + pred_h / 2.0

        tgt_x1 = tgt_xc - tgt_w / 2.0
        tgt_y1 = tgt_yc - tgt_h / 2.0
        tgt_x2 = tgt_xc + tgt_w / 2.0
        tgt_y2 = tgt_yc + tgt_h / 2.0

        # Intersection
        inter_x1 = torch.maximum(pred_x1, tgt_x1)
        inter_y1 = torch.maximum(pred_y1, tgt_y1)
        inter_x2 = torch.minimum(pred_x2, tgt_x2)
        inter_y2 = torch.minimum(pred_y2, tgt_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        # Areas
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h

        union_area = pred_area + tgt_area - inter_area
        iou = inter_area / (union_area + self.eps)

        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss