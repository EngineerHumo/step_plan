"""Permutation-invariant matching utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch


def dice_cost(pred_logits: torch.Tensor, gt_masks: torch.Tensor, roi: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Compute (1 - dice) between predictions and ground-truth masks."""

    B, K, H, W = pred_logits.shape
    Kgt = gt_masks.shape[1]
    pred = pred_logits.sigmoid().view(B, K, -1)
    gt = gt_masks.view(B, Kgt, -1).clamp(0, 1)
    if roi is not None:
        roi_flat = roi.view(B, 1, -1)
        pred = pred * roi_flat
        gt = gt * roi_flat
    inter = torch.einsum("bki,bgi->bkg", pred, gt)
    pred_sum = pred.sum(-1, keepdim=True)
    gt_sum = gt.sum(-1).unsqueeze(1)
    dice = (2 * inter + eps) / (pred_sum + gt_sum + eps)
    return 1.0 - dice


def bce_cost(pred_logits: torch.Tensor, gt_masks: torch.Tensor, roi: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    B, K, H, W = pred_logits.shape
    Kgt = gt_masks.shape[1]
    pred = pred_logits.sigmoid().view(B, K, -1)
    gt = gt_masks.view(B, Kgt, -1).clamp(0, 1)
    if roi is not None:
        roi_flat = roi.view(B, 1, -1)
        pred = pred * roi_flat
        gt = gt * roi_flat
        normalizer = roi_flat.sum(-1, keepdim=True).clamp_min(1.0)
    else:
        normalizer = torch.tensor(H * W, device=pred.device, dtype=pred.dtype)
    loss_pos = torch.einsum("bki,bgi->bkg", torch.log(pred + eps), gt)
    loss_neg = torch.einsum("bki,bgi->bkg", torch.log(1.0 - pred + eps), 1.0 - gt)
    bce = -(loss_pos + loss_neg) / normalizer
    return bce


def hungarian_match(cost: torch.Tensor, valid_gt_mask: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Greedy approximate Hungarian matching implemented in PyTorch."""

    matches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    B, K, Kmax = cost.shape
    for b in range(B):
        cols = valid_gt_mask[b].nonzero(as_tuple=False).flatten()
        if cols.numel() == 0:
            matches.append((torch.empty(0, dtype=torch.long, device=cost.device), torch.empty(0, dtype=torch.long, device=cost.device)))
            continue
        cost_slice = cost[b][:, cols]
        remaining_rows = torch.arange(K, device=cost.device).tolist()
        remaining_cols = cols.tolist()
        row_match: List[int] = []
        col_values: List[int] = []
        while remaining_rows and remaining_cols:
            min_val = None
            sel_row = None
            sel_idx = None
            for r in remaining_rows:
                row_vals = cost_slice[r]
                for c_idx, col_val in enumerate(remaining_cols):
                    val = row_vals[c_idx].item()
                    if min_val is None or val < min_val:
                        min_val = val
                        sel_row = r
                        sel_idx = c_idx
            if sel_row is None or sel_idx is None:
                break
            row_match.append(sel_row)
            col_values.append(remaining_cols[sel_idx])
            remaining_rows.remove(sel_row)
            remaining_cols.pop(sel_idx)
        rows_tensor = torch.tensor(row_match, dtype=torch.long, device=cost.device)
        cols_tensor = cols.new_tensor(col_values, dtype=torch.long) if col_values else torch.empty(0, dtype=torch.long, device=cost.device)
        matches.append((rows_tensor, cols_tensor))
    return matches


__all__ = ["dice_cost", "bce_cost", "hungarian_match"]
