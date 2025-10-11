"""Loss functions for permutation-invariant surgical planning."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if mask is not None:
        pred = pred * mask
        target = target * mask
    inter = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (2.0 * inter + eps) / denom


def bce_loss_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        weight = mask
    else:
        weight = None
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
    return loss.mean(dim=(2, 3))


def overlap_penalty(pred_prob: torch.Tensor, roi: torch.Tensor | None = None) -> torch.Tensor:
    B, K, H, W = pred_prob.shape
    flat = pred_prob.view(B, K, -1)
    if roi is not None:
        roi_flat = roi.view(B, 1, -1)
        flat = flat * roi_flat
        normalizer = roi_flat.sum(-1).clamp_min(1.0)
    else:
        normalizer = torch.tensor(H * W, device=flat.device, dtype=flat.dtype)
    penalty = torch.zeros(B, device=flat.device, dtype=flat.dtype)
    pair_count = max(1, K * (K - 1) // 2)
    for i in range(K):
        for j in range(i + 1, K):
            overlap = (flat[:, i] * flat[:, j]).sum(-1) / normalizer
            penalty += overlap
    return penalty / pair_count


def tv_smoothness(pred_prob: torch.Tensor) -> torch.Tensor:
    dx = pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]
    dy = pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(1)
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


def boundary_alignment_loss(mask_logits: torch.Tensor, image: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    prob_sum = mask_logits.sigmoid().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    pred_edge = sobel_edges(prob_sum)
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image
    gray = (gray - gray.mean(dim=(2, 3), keepdim=True)) / (gray.std(dim=(2, 3), keepdim=True) + 1e-6)
    img_edge = sobel_edges(gray)
    return (pred_edge - img_edge).abs().mean() * weight


def forbidden_overlap_loss(pred_prob: torch.Tensor, forbidden_mask: torch.Tensor) -> torch.Tensor:
    return (pred_prob * forbidden_mask).mean()


def area_prior_loss(pred_prob: torch.Tensor, matched_gt: torch.Tensor, roi: torch.Tensor | None = None) -> torch.Tensor:
    if roi is not None:
        pred_prob = pred_prob * roi
        matched_gt = matched_gt * roi
        normalizer = roi.sum(dim=(2, 3)).clamp_min(1.0)
    else:
        normalizer = torch.tensor(pred_prob.shape[-1] * pred_prob.shape[-2], device=pred_prob.device, dtype=pred_prob.dtype)
    pred_area = pred_prob.sum(dim=(2, 3)) / normalizer
    gt_area = matched_gt.sum(dim=(2, 3)) / normalizer
    return (pred_area - gt_area).abs().mean()


def existence_losses(exist_logits: torch.Tensor, match_rows: Sequence[torch.Tensor], K_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, K = exist_logits.shape
    targets = torch.zeros_like(exist_logits, dtype=torch.float32)
    for b in range(B):
        rows = match_rows[b]
        if rows.numel() > 0:
            targets[b, rows] = 1.0
    exist_ce = F.binary_cross_entropy_with_logits(exist_logits, targets)
    pred_count = torch.sigmoid(exist_logits).sum(dim=1)
    card = F.mse_loss(pred_count, K_gt.to(pred_count.dtype))
    return exist_ce, card


__all__ = [
    "dice_loss",
    "bce_loss_logits",
    "overlap_penalty",
    "tv_smoothness",
    "sobel_edges",
    "boundary_alignment_loss",
    "forbidden_overlap_loss",
    "area_prior_loss",
    "existence_losses",
]
