"""Training script for permutation-invariant surgical sub-region segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import cv2

from config import Config
from dataset import (
    SurgicalPlanningDataset,
    build_synthetic_items,
    collate_fn,
    load_items,
)
from losses import (
    area_prior_loss,
    asymmetric_bce_loss,
    boundary_alignment_loss,
    dice_loss,
    existence_losses,
    matched_false_positive_loss,
    overlap_penalty,
    query_cluster_separation_loss,
    query_compactness_loss,
    query_diversity_loss,
    query_kernel_diversity_loss,
    tversky_loss,
    unmatched_spillover_loss,
    tv_smoothness,
)
from matcher import bce_cost, dice_cost, hungarian_match
from model_parts import InputFusion, MaskDecoder, PixelDecoder, SegFormerBackbone
from utils.seed import set_global_seed

try:
    from visdom import Visdom
except Exception:  # pragma: no cover - optional dependency
    Visdom = None


class VisdomLogger:
    """Utility for streaming training visuals to a Visdom dashboard."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.enabled = cfg.visdom_enabled and Visdom is not None
        self.vis: Visdom | None = None
        self._wins: Dict[str, str] = {}
        self.global_step = 0
        if not self.enabled:
            return
        try:
            self.vis = Visdom(server=cfg.visdom_server, port=cfg.visdom_port, env=cfg.visdom_env)
            if not self.vis.check_connection():
                print("[Visdom] Connection failed, disabling visualisation.")
                self.vis = None
                self.enabled = False
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[Visdom] Initialisation error: {exc}")
            self.vis = None
            self.enabled = False

    def _to_grid(self, tensor: torch.Tensor, normalize: bool = True, nrow: int | None = None) -> np.ndarray | None:
        if tensor.numel() == 0:
            return None
        data = tensor.detach().cpu().float()
        if data.dim() == 3:
            data = data.unsqueeze(1)
        data = data[: self.cfg.visdom_max_samples]
        if data.size(0) == 0:
            return None
        if normalize:
            flat = data.view(data.size(0), -1)
            min_vals = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            max_vals = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            data = (data - min_vals) / (max_vals - min_vals + 1e-6)
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)
        grid = make_grid(
            data,
            nrow=nrow or max(1, min(data.size(0), self.cfg.visdom_max_samples)),
            padding=2,
        )
        return grid.cpu().numpy()

    def _queries_to_grid(
        self,
        tensor: torch.Tensor,
        roi: torch.Tensor | None = None,
    ) -> np.ndarray | None:
        if tensor.numel() == 0:
            return None
        data = tensor.detach().cpu().float()
        B, K, H, W = data.shape
        max_samples = min(B, self.cfg.visdom_max_samples)
        if max_samples == 0:
            return None
        data = data[:max_samples]
        if roi is not None:
            roi = roi.detach().cpu().float()[:max_samples]
            data = data * roi
        data = data.view(max_samples * K, 1, H, W)
        flat = data.view(data.size(0), -1)
        min_vals = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        data = (data - min_vals) / (max_vals - min_vals + 1e-6)
        data = data.repeat(1, 3, 1, 1)
        grid = make_grid(data, nrow=self.cfg.num_queries, padding=2)
        return grid.cpu().numpy()

    def _show(self, key: str, img: np.ndarray | None, title: str) -> None:
        if not self.enabled or self.vis is None or img is None:
            return
        opts = {"title": f"{title} (step {self.global_step})"}
        win = self._wins.get(key)
        self._wins[key] = self.vis.image(img, win=win, opts=opts)

    def log_losses(self, losses: Dict[str, float]) -> None:
        if not self.enabled or self.vis is None:
            return
        total = losses.get("total")
        if total is None:
            return
        x = np.array([self.global_step], dtype=np.float32)
        y = np.array([total], dtype=np.float32)
        win = self._wins.get("total_loss")
        update = None if win is None else "append"
        self._wins["total_loss"] = self.vis.line(
            X=x,
            Y=y,
            win=win,
            update=update,
            opts={"title": "Train/Total Loss", "xlabel": "step", "ylabel": "loss"},
        )

    
    def log_batch(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_prob: torch.Tensor,
        roi: torch.Tensor,
        split: str = "Train",
    ) -> None:
        if not self.enabled or self.vis is None:
            return
        try:
            inputs_grid = self._to_grid(images, normalize=True)
            gt_grid = self._queries_to_grid(gt_masks.float(), roi=roi)
            pred_grid = self._queries_to_grid(pred_prob, roi=roi)
            combined_gt = gt_masks.float().max(dim=1, keepdim=True).values
            combined_pred = pred_prob.max(dim=1, keepdim=True).values
            gt_combined_grid = self._to_grid(combined_gt, normalize=True)
            pred_combined_grid = self._to_grid(combined_pred, normalize=True)

            prefix = split.lower()
            title_prefix = split.capitalize()

            self._show(f"{prefix}_inputs", inputs_grid, f"{title_prefix}/Input")
            self._show(
                f"{prefix}_labels_queries",
                gt_grid,
                f"{title_prefix}/GT_Queries (4x6)",
            )
            self._show(
                f"{prefix}_outputs_queries",
                pred_grid,
                f"{title_prefix}/Pred_Queries (4x6)",
            )
            self._show(
                f"{prefix}_labels_combined",
                gt_combined_grid,
                f"{title_prefix}/Label_Combined",
            )
            self._show(
                f"{prefix}_outputs_combined",
                pred_combined_grid,
                f"{title_prefix}/Output_Combined",
            )

            if prefix == "train":
                self.global_step += 1
        except Exception as exc:
            print(f"[Visdom] Logging error: {exc}")
            self.enabled = False
            self.vis = None


class FullModel(nn.Module):
    """Complete model that fuses inputs, encodes, and decodes masks."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.fuse = InputFusion(cfg.in_channels, cfg.aux_mask_channels, out_c=3)
        self.backbone = SegFormerBackbone(cfg)
        self.adapters = nn.ModuleList(
            [nn.Conv2d(ch, cfg.embed_dim, kernel_size=1) for ch in self.backbone.out_channels]
        )
        self.pixel_decoder = PixelDecoder([cfg.embed_dim] * len(self.adapters), cfg.embed_dim)
        self.mask_decoder = MaskDecoder(cfg)

    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        fused = self.fuse(image, aux)
        feats = self.backbone(fused)
        feats = [adapter(feat) for adapter, feat in zip(self.adapters, feats)]
        pix_embed = self.pixel_decoder(feats)
        mask_logits, exist_logits, lowres, kernels = self.mask_decoder(pix_embed)
        return mask_logits, exist_logits, {"pix": pix_embed, "lowres": lowres, "kernels": kernels}


def _select_device(preferred: str) -> str:
    if preferred == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred


def _get_roi_mask(aux: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Return a valid ROI mask, falling back to all-ones when empty."""
    #print(cfg.roi_channel_index)
    if aux.size(1) == 0:
        shape = (aux.size(0), 1, aux.size(-2), aux.size(-1))
        return torch.ones(shape, dtype=torch.float32, device=aux.device)

    roi = aux[:, cfg.roi_channel_index : cfg.roi_channel_index + 1].clone().float()
    flat = (roi > 0.5).flatten(2).sum(-1)
    #print(roi.max())
    #print(roi.shape)
    #print(flat)
    #empty = flat == 0
    #if empty.any():
    #    roi[empty] = 1.0
    return roi


def improved_cost_calculation(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    roi: torch.Tensor | None,
) -> torch.Tensor:
    B, K, H, W = mask_logits.shape
    Kgt = gt_masks.shape[1]

    dice_cost_val = dice_cost(mask_logits, gt_masks, roi=roi)
    bce_cost_val = bce_cost(mask_logits, gt_masks, roi=roi)

    diversity_cost = mask_logits.new_zeros(B, K, Kgt)
    if K > 1:
        pred_masks = mask_logits.sigmoid().view(B, K, -1)
        if roi is not None:
            roi_flat = roi.view(B, 1, -1)
            pred_masks = pred_masks * roi_flat
        for b in range(B):
            others = pred_masks[b]
            for k in range(K):
                other_indices = [i for i in range(K) if i != k]
                if not other_indices:
                    continue
                other_mean = others[other_indices].mean(dim=0)
                similarity = F.cosine_similarity(pred_masks[b, k], other_mean, dim=0)
                diversity_cost[b, k] = similarity.clamp_min(0.0) * 0.1

    cost = 0.6 * dice_cost_val + 0.4 * bce_cost_val + diversity_cost
    return cost


def train_one_epoch(
    model: FullModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: Config,
    vis_logger: Optional[VisdomLogger] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(cfg.device)
        aux = batch["aux"].to(cfg.device)
        gt_masks = batch["gt_masks"].to(cfg.device)
        gt_valid = batch["gt_valid"].to(cfg.device)
        K_gt = batch["K_gt"].to(cfg.device)

        roi = _get_roi_mask(aux, cfg).to(image.dtype)

        mask_logits, exist_logits, decoder_out = model(image, aux)
        pred_prob = mask_logits.sigmoid() * roi

        cost = improved_cost_calculation(mask_logits, gt_masks, roi)
        matches = hungarian_match(cost, gt_valid)

        B, K, H, W = mask_logits.shape
        matched_gt = torch.zeros_like(mask_logits)
        matched_mask = torch.zeros((B, K), dtype=torch.bool, device=mask_logits.device)
        for b, (rows, cols) in enumerate(matches):
            if rows.numel() == 0:
                continue
            matched_gt[b, rows] = gt_masks[b, cols]
            matched_mask[b, rows] = True

        matched_gt = matched_gt * roi

        matched_dice_vals = tversky_loss(pred_prob, matched_gt, mask=roi, alpha=0.7, beta=0.3)
        matched_bce_vals = asymmetric_bce_loss(
            mask_logits,
            matched_gt,
            mask=roi,
            pos_weight=1.0,
            neg_weight=1.5,
        )
        if matched_mask.any():
            matched_dice_raw = matched_dice_vals[matched_mask].mean()
            matched_bce_raw = matched_bce_vals[matched_mask].mean()
        else:
            matched_dice_raw = mask_logits.new_tensor(0.0)
            matched_bce_raw = mask_logits.new_tensor(0.0)

        overlap_raw = overlap_penalty(pred_prob, roi=roi).mean()
        overlap_weight = cfg.w_overlap * 1.5
        overlap_weighted = overlap_weight * overlap_raw
        tv_raw = tv_smoothness(pred_prob)
        tv_weighted = cfg.w_tv * tv_raw
        boundary_raw = boundary_alignment_loss(mask_logits.detach(), image)
        boundary_weighted = cfg.w_boundary * boundary_raw
        area_raw = area_prior_loss(pred_prob, matched_gt, roi=roi)
        area_weighted = cfg.w_area * area_raw
        exist_ce_raw, card_raw = existence_losses(exist_logits, [m[0] for m in matches], K_gt)
        exist_weighted = cfg.w_exist_ce * exist_ce_raw
        card_weighted = cfg.w_cardinality * card_raw

        matched_fp_raw = matched_false_positive_loss(
            pred_prob,
            matched_gt,
            matched_mask,
            roi=roi,
            exist_logits=exist_logits,
        )
        matched_fp_weighted = cfg.w_matched_fp * matched_fp_raw

        diversity_raw = query_diversity_loss(pred_prob, mask=roi)
        diversity_weighted = 0.1 * diversity_raw
        kernel_div_raw = query_kernel_diversity_loss(decoder_out["kernels"])
        kernel_div_weighted = 0.05 * kernel_div_raw

        unmatched_mask = ~matched_mask
        unmatched_dice_raw_list: List[torch.Tensor] = []
        unmatched_bce_raw_list: List[torch.Tensor] = []
        unmatched_overlap_raw_list: List[torch.Tensor] = []
        roi_area = roi.view(B, -1).sum(dim=1).clamp_min(1.0)
        for b in range(B):
            unmatched_indices = torch.nonzero(unmatched_mask[b], as_tuple=False).squeeze(1)
            if unmatched_indices.numel() == 0:
                continue
            valid_cols = torch.nonzero(gt_valid[b], as_tuple=False).squeeze(1)
            matched_indices = torch.nonzero(matched_mask[b], as_tuple=False).squeeze(1)
            for idx in unmatched_indices:
                exist_prob = torch.sigmoid(exist_logits[b, idx])
                roi_slice = roi[b : b + 1]
                if valid_cols.numel() > 0:
                    costs = cost[b, idx, valid_cols]
                    if costs.numel() > 0:
                        best_col = valid_cols[costs.argmin()]
                        pred_slice = pred_prob[b : b + 1, idx : idx + 1]
                        logit_slice = mask_logits[b : b + 1, idx : idx + 1]
                        gt_slice = gt_masks[b : b + 1, best_col : best_col + 1] * roi_slice
                        dice_val = tversky_loss(pred_slice, gt_slice, mask=roi_slice, alpha=0.7, beta=0.3).mean()
                        bce_val = asymmetric_bce_loss(
                            logit_slice,
                            gt_slice,
                            mask=roi_slice,
                            pos_weight=1.0,
                            neg_weight=1.5,
                        ).mean()
                        unmatched_dice_raw_list.append(exist_prob * dice_val)
                        unmatched_bce_raw_list.append(exist_prob * bce_val)
                if matched_indices.numel() > 0:
                    matched_probs = pred_prob[b, matched_indices]
                    overlap_vals = (
                        pred_prob[b, idx : idx + 1] * matched_probs
                    ).view(matched_probs.size(0), -1).sum(dim=1)
                    overlap_mean = overlap_vals.mean() / roi_area[b]
                    unmatched_overlap_raw_list.append(exist_prob * overlap_mean)

        if unmatched_dice_raw_list:
            unmatched_dice_raw = torch.stack(unmatched_dice_raw_list).mean()
        else:
            unmatched_dice_raw = mask_logits.new_tensor(0.0)
        unmatched_dice_weighted = cfg.w_unmatched_dice * unmatched_dice_raw

        if unmatched_bce_raw_list:
            unmatched_bce_raw = torch.stack(unmatched_bce_raw_list).mean()
        else:
            unmatched_bce_raw = mask_logits.new_tensor(0.0)
        unmatched_bce_weighted = cfg.w_unmatched_bce * unmatched_bce_raw

        if unmatched_overlap_raw_list:
            unmatched_overlap_raw = torch.stack(unmatched_overlap_raw_list).mean()
        else:
            unmatched_overlap_raw = mask_logits.new_tensor(0.0)
        unmatched_overlap_weighted = cfg.w_overlap * unmatched_overlap_raw

        unmatched_spill_raw = unmatched_spillover_loss(
            pred_prob,
            unmatched_mask,
            gt_masks,
            roi=roi,
            exist_logits=exist_logits,
        )
        unmatched_spill_weighted = cfg.w_unmatched_spill * unmatched_spill_raw

        matched_dice_weighted = cfg.w_dice * matched_dice_raw
        matched_bce_weighted = cfg.w_bce * matched_bce_raw

        cluster_raw = query_cluster_separation_loss(
            pred_prob,
            matches,
            roi=roi,
            exist_logits=exist_logits,
        )
        cluster_weighted = cfg.w_query_cluster * cluster_raw

        compact_raw = query_compactness_loss(
            pred_prob,
            roi=roi,
            exist_logits=exist_logits,
        )
        compact_weighted = cfg.w_query_compact * compact_raw

        loss = (
            matched_dice_weighted
            + matched_bce_weighted
            + matched_fp_weighted
            + overlap_weighted
            #+ tv_weighted
            #+ boundary_weighted
            #+ area_weighted
            + exist_weighted
            + card_weighted
            #+ unmatched_dice_weighted
            #+ unmatched_bce_weighted
            #+ unmatched_overlap_weighted
            #+ unmatched_spill_weighted
            #+ diversity_weighted
            #+ kernel_div_weighted
            + cluster_weighted
            #+ compact_weighted
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        loss_report = [
            f"matched_dice: raw={matched_dice_raw.item():.4f}, weighted={matched_dice_weighted.item():.4f}",
            f"matched_bce: raw={matched_bce_raw.item():.4f}, weighted={matched_bce_weighted.item():.4f}",
            f"unmatched_dice: raw={unmatched_dice_raw.item():.4f}, weighted={unmatched_dice_weighted.item():.4f}",
            f"unmatched_bce: raw={unmatched_bce_raw.item():.4f}, weighted={unmatched_bce_weighted.item():.4f}",
            f"unmatched_overlap: raw={unmatched_overlap_raw.item():.4f}, weighted={unmatched_overlap_weighted.item():.4f}",
            f"unmatched_spill: raw={unmatched_spill_raw.item():.4f}, weighted={unmatched_spill_weighted.item():.4f}",
            f"overlap_penalty: raw={overlap_raw.item():.4f}, weighted={overlap_weighted.item():.4f}",
            f"tv: raw={tv_raw.item():.4f}, weighted={tv_weighted.item():.4f}",
            f"boundary: raw={boundary_raw.item():.4f}, weighted={boundary_weighted.item():.4f}",
            f"area: raw={area_raw.item():.4f}, weighted={area_weighted.item():.4f}",
            f"exist_ce: raw={exist_ce_raw.item():.4f}, weighted={exist_weighted.item():.4f}",
            f"cardinality: raw={card_raw.item():.4f}, weighted={card_weighted.item():.4f}",
            f"matched_fp: raw={matched_fp_raw.item():.4f}, weighted={matched_fp_weighted.item():.4f}",
            f"query_diversity: raw={diversity_raw.item():.4f}, weighted={diversity_weighted.item():.4f}",
            f"kernel_diversity: raw={kernel_div_raw.item():.4f}, weighted={kernel_div_weighted.item():.4f}",
            f"cluster: raw={cluster_raw.item():.4f}, weighted={cluster_weighted.item():.4f}",
            f"compact: raw={compact_raw.item():.4f}, weighted={compact_weighted.item():.4f}",
        ]
        print("Loss breakdown:")
        for entry in loss_report:
            print("  " + entry)
        print(f"  total: {loss.item():.4f}")
        print(f"Cost matrix range: [{cost.min():.3f}, {cost.max():.3f}]")
        for b, (rows, cols) in enumerate(matches):
            if rows.numel() > 0:
                print(f"Batch {b}: {len(rows)} matches - queries {rows.tolist()} -> targets {cols.tolist()}")
            else:
                print(f"Batch {b}: No matches")
        pred_std = pred_prob.std(dim=1).mean()
        kernel_std = decoder_out["kernels"].std(dim=1).mean()
        print(f"Prediction std across queries: {pred_std:.4f}")
        print(f"Kernel std across queries: {kernel_std:.4f}")
        if vis_logger is not None:
            loss_dict = {
                "total": float(loss.item()),
                "matched_dice_raw": float(matched_dice_raw.item()),
                "matched_dice_weighted": float(matched_dice_weighted.item()),
                "matched_bce_raw": float(matched_bce_raw.item()),
                "matched_bce_weighted": float(matched_bce_weighted.item()),
                "unmatched_dice_raw": float(unmatched_dice_raw.item()),
                "unmatched_dice_weighted": float(unmatched_dice_weighted.item()),
                "unmatched_bce_raw": float(unmatched_bce_raw.item()),
                "unmatched_bce_weighted": float(unmatched_bce_weighted.item()),
                "unmatched_overlap_raw": float(unmatched_overlap_raw.item()),
                "unmatched_overlap_weighted": float(unmatched_overlap_weighted.item()),
                "overlap_raw": float(overlap_raw.item()),
                "overlap_weighted": float(overlap_weighted.item()),
                "tv_raw": float(tv_raw.item()),
                "tv_weighted": float(tv_weighted.item()),
                "boundary_raw": float(boundary_raw.item()),
                "boundary_weighted": float(boundary_weighted.item()),
                "area_raw": float(area_raw.item()),
                "area_weighted": float(area_weighted.item()),
                "exist_ce_raw": float(exist_ce_raw.item()),
                "exist_ce_weighted": float(exist_weighted.item()),
                "card_raw": float(card_raw.item()),
                "card_weighted": float(card_weighted.item()),
                "diversity_raw": float(diversity_raw.item()),
                "diversity_weighted": float(diversity_weighted.item()),
                "kernel_div_raw": float(kernel_div_raw.item()),
                "kernel_div_weighted": float(kernel_div_weighted.item()),
            }
            vis_logger.log_losses(loss_dict)
            vis_logger.log_batch(image, gt_masks, pred_prob, roi, split="Train")
            #vis_logger.log_batch(image, gt_masks[:,0,:,:], pred_prob[:,0,:,:],gt_masks[:,1,:,:], pred_prob[:,0,:,:],gt_masks[:,0,:,:], pred_prob[:,0,:,:],gt_masks[:,0,:,:], pred_prob[:,0,:,:])
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: FullModel,
    loader: DataLoader,
    cfg: Config,
    vis_logger: Optional[VisdomLogger] = None,
) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    matched_total = 0
    logged_visuals = False
    for batch in loader:
        image = batch["image"].to(cfg.device)
        aux = batch["aux"].to(cfg.device)
        gt_masks = batch["gt_masks"].to(cfg.device)
        gt_valid = batch["gt_valid"].to(cfg.device)

        roi = _get_roi_mask(aux, cfg).to(image.dtype)
        mask_logits, exist_logits, _ = model(image, aux)
        pred_prob = mask_logits.sigmoid() * roi
        cost = dice_cost(mask_logits, gt_masks, roi=roi)
        matches = hungarian_match(cost, gt_valid)

        matched_gt = torch.zeros_like(mask_logits)
        matched_mask = torch.zeros((mask_logits.shape[0], mask_logits.shape[1]), dtype=torch.bool, device=mask_logits.device)
        for b, (rows, cols) in enumerate(matches):
            if rows.numel() == 0:
                continue
            matched_gt[b, rows] = gt_masks[b, cols]
            matched_mask[b, rows] = True

        matched_gt = matched_gt * roi
        dice_vals = 1.0 - dice_loss(pred_prob, matched_gt, mask=roi)
        if matched_mask.any():
            total_dice += float(dice_vals[matched_mask].mean().item())
            matched_total += 1
        if vis_logger is not None and not logged_visuals:
            vis_logger.log_batch(image, gt_masks, pred_prob, roi, split="Val")
            logged_visuals = True
    if matched_total == 0:
        return {"dice": 0.0}
    return {"dice": total_dice / matched_total}


@torch.no_grad()
def save_validation_predictions(
    model: FullModel,
    loader: DataLoader,
    cfg: Config,
    epoch: int,
    output_root: Path | None = None,
) -> None:
    """Save validation predictions for visual inspection."""

    model.eval()
    root = Path("runs/output") if output_root is None else Path(output_root)
    epoch_dir = root / f"epoch_{epoch:04d}"
    if epoch_dir.exists():
        shutil.rmtree(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    for batch in loader:
        images = batch["image"].to(cfg.device)
        aux = batch["aux"].to(cfg.device)
        roi = _get_roi_mask(aux, cfg).to(images.dtype)

        mask_logits, _, _ = model(images, aux)
        pred_prob = mask_logits.sigmoid() * roi

        meta_list = batch.get("meta", [{} for _ in range(images.size(0))])
        for b in range(images.size(0)):
            meta = meta_list[b] if b < len(meta_list) else {}
            image_path = meta.get("image_path")
            case_id = meta.get("case_id")
            if not case_id:
                if image_path is not None:
                    case_id = Path(image_path).stem
                else:
                    case_id = f"sample_{sample_idx:04d}"

            case_dir = epoch_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            probs = pred_prob[b].detach().cpu().numpy()
            combined = probs.max(axis=0)

            for q, prob in enumerate(probs):
                out_path = case_dir / f"query_{q:02d}.png"
                cv2.imwrite(str(out_path), (prob * 255).clip(0, 255).astype(np.uint8))

            combined_path = case_dir / "combined.png"
            cv2.imwrite(
                str(combined_path),
                (combined * 255).clip(0, 255).astype(np.uint8),
            )

            sample_idx += 1


def save_checkpoint(path: Path, model: FullModel, optimizer: optim.Optimizer, epoch: int, cfg: Config, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": asdict(cfg),
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: Path, model: FullModel, optimizer: Optional[optim.Optimizer], cfg: Config) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", 0.0))
    return start_epoch, best_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surgical sub-region segmentation model")
    parser.add_argument("--train_items", type=str, default=None, help="Path to training items JSON")
    parser.add_argument("--val_items", type=str, default=None, help="Path to validation items JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--img_size", type=int, nargs=2, default=None, help="Image size H W")
    parser.add_argument("--num_queries", type=int, default=None, help="Number of queries")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=1, help="Device to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data for smoke testing")
    parser.add_argument("--synthetic_samples", type=int, default=32, help="Number of synthetic samples")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--visdom", action="store_true", help="Enable Visdom visualisation")
    parser.add_argument("--visdom_env", type=str, default=None, help="Visdom environment name")
    parser.add_argument("--visdom_server", type=str, default=None, help="Visdom server address")
    parser.add_argument("--visdom_port", type=int, default=None, help="Visdom server port")
    parser.add_argument(
        "--visdom_max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to visualise per batch",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=None,
        help="Number of epochs between validation runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.img_size is not None:
        cfg.img_size = (int(args.img_size[0]), int(args.img_size[1]))
    if args.num_queries is not None:
        cfg.num_queries = int(args.num_queries)
    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.max_epochs = int(args.epochs)
    if args.val_interval is not None and args.val_interval > 0:
        cfg.val_interval = int(args.val_interval)
    if args.visdom:
        cfg.visdom_enabled = True
    if args.visdom_env is not None:
        cfg.visdom_env = args.visdom_env
    if args.visdom_server is not None:
        cfg.visdom_server = args.visdom_server
    if args.visdom_port is not None:
        cfg.visdom_port = int(args.visdom_port)
    if args.visdom_max_samples is not None:
        cfg.visdom_max_samples = max(1, int(args.visdom_max_samples))
    cfg.device = _select_device(cfg.device)

    set_global_seed(args.seed)

    if args.synthetic:
        synth_root = Path(cfg.save_dir) / "synthetic_data"
        items = build_synthetic_items(synth_root, args.synthetic_samples, cfg)
        split = int(len(items) * 0.8)
        train_items = items[:split]
        val_items = items[split:]
    else:
        if args.train_items is None:
            raise ValueError("--train_items must be provided unless --synthetic is set")
        train_items = load_items(args.train_items)
        val_items = load_items(args.val_items) if args.val_items is not None else []

    train_dataset = SurgicalPlanningDataset(train_items, cfg, augment=not args.synthetic)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, cfg),
    )
    if val_items:
        val_dataset = SurgicalPlanningDataset(val_items, cfg, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, cfg),
        )
    else:
        val_loader = None

    model = FullModel(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    best_metric = 0.0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        start_epoch, best_metric = load_checkpoint(ckpt_path, model, optimizer, cfg)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis_logger = VisdomLogger(cfg) if cfg.visdom_enabled else None

    for epoch in range(start_epoch, cfg.max_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, vis_logger=vis_logger)
        elapsed = time.time() - start_time
        val_metrics: Optional[Dict[str, float]] = None
        should_validate = (
            val_loader is not None
            and (((epoch + 1) % cfg.val_interval == 0) or (epoch == cfg.max_epochs - 1))
        )
        if should_validate and val_loader is not None:
            val_metrics = evaluate(model, val_loader, cfg, vis_logger=vis_logger)
            save_validation_predictions(model, val_loader, cfg, epoch + 1)
            if val_metrics["dice"] > best_metric:
                best_metric = val_metrics["dice"]
                save_checkpoint(
                    save_dir / "best.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    cfg,
                    best_metric,
                )
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch + 1, cfg, best_metric)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_dice": None if val_metrics is None else val_metrics.get("dice", 0.0),
                    "elapsed_sec": elapsed,
                }
            )
        )


if __name__ == "__main__":
    main()
