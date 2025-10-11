"""Evaluation metrics for instance segmentation."""

from __future__ import annotations

from typing import List

import torch
from scipy.ndimage import distance_transform_edt


def binarize(mask_probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask_probs >= threshold).to(mask_probs.dtype)


def dice_per_instance(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * gt).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    return (2.0 * inter + eps) / (union + eps)


def aggregated_jaccard(pred_instances: List[torch.Tensor], gt_instances: List[torch.Tensor], eps: float = 1e-6) -> float:
    if not gt_instances:
        return 1.0 if not pred_instances else 0.0
    pred_stack = torch.stack(pred_instances, dim=0) if pred_instances else torch.zeros((0, *gt_instances[0].shape), dtype=torch.float32)
    gt_stack = torch.stack(gt_instances, dim=0)
    inter = torch.einsum("khw,ghw->kg", pred_stack, gt_stack)
    union = pred_stack.sum(dim=(1, 2)).unsqueeze(1) + gt_stack.sum(dim=(1, 2)) - inter
    if pred_instances:
        match = inter / (union + eps)
        best = match.max(dim=0).values
        return float(best.sum().item() / gt_stack.shape[0])
    return 0.0


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, tolerance: int = 2) -> float:
    pred_np = pred.cpu().numpy().astype(bool)
    gt_np = gt.cpu().numpy().astype(bool)
    pred_dt = distance_transform_edt(~pred_np)
    gt_dt = distance_transform_edt(~gt_np)
    pred_boundary = pred_dt <= tolerance
    gt_boundary = gt_dt <= tolerance
    precision = (pred_boundary & gt_np).sum() / max(1, pred_boundary.sum())
    recall = (gt_boundary & pred_np).sum() / max(1, gt_boundary.sum())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


__all__ = [
    "binarize",
    "dice_per_instance",
    "aggregated_jaccard",
    "boundary_f1",
]
