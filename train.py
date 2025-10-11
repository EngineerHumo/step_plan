"""Training script for permutation-invariant surgical sub-region segmentation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    SurgicalPlanningDataset,
    build_synthetic_items,
    collate_fn,
    load_items,
)
from losses import (
    area_prior_loss,
    bce_loss_logits,
    boundary_alignment_loss,
    dice_loss,
    existence_losses,
    forbidden_overlap_loss,
    overlap_penalty,
    tv_smoothness,
)
from matcher import bce_cost, dice_cost, hungarian_match
from model_parts import InputFusion, MaskDecoder, PixelDecoder, SegFormerBackbone
from utils.seed import set_global_seed


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
        mask_logits, exist_logits, lowres = self.mask_decoder(pix_embed)
        return mask_logits, exist_logits, {"pix": pix_embed, "lowres": lowres}


def _select_device(preferred: str) -> str:
    if preferred == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred


def train_one_epoch(model: FullModel, loader: DataLoader, optimizer: optim.Optimizer, cfg: Config) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(cfg.device)
        aux = batch["aux"].to(cfg.device)
        gt_masks = batch["gt_masks"].to(cfg.device)
        gt_valid = batch["gt_valid"].to(cfg.device)
        K_gt = batch["K_gt"].to(cfg.device)

        roi = aux[:, cfg.roi_channel_index : cfg.roi_channel_index + 1]
        forbidden = aux[:, 1:3].sum(1, keepdim=True).clamp(max=1.0)

        mask_logits, exist_logits, _ = model(image, aux)
        pred_prob = mask_logits.sigmoid()

        C_dice = dice_cost(mask_logits, gt_masks, roi=roi)
        C_bce = bce_cost(mask_logits, gt_masks, roi=roi)
        cost = 0.6 * C_dice + 0.4 * C_bce
        matches = hungarian_match(cost, gt_valid)

        B, K, H, W = mask_logits.shape
        matched_gt = torch.zeros_like(mask_logits)
        matched_mask = torch.zeros((B, K), dtype=torch.bool, device=mask_logits.device)
        for b, (rows, cols) in enumerate(matches):
            if rows.numel() == 0:
                continue
            matched_gt[b, rows] = gt_masks[b, cols]
            matched_mask[b, rows] = True

        dice_vals = dice_loss(pred_prob, matched_gt, mask=roi)
        bce_vals = bce_loss_logits(mask_logits, matched_gt, mask=roi)
        if matched_mask.any():
            dice_term = dice_vals[matched_mask].mean()
            bce_term = bce_vals[matched_mask].mean()
        else:
            dice_term = torch.tensor(0.0, device=mask_logits.device)
            bce_term = torch.tensor(0.0, device=mask_logits.device)

        overlap = overlap_penalty(pred_prob, roi=roi).mean()
        tv = tv_smoothness(pred_prob)
        boundary = boundary_alignment_loss(mask_logits.detach(), image)
        forbid_pen = forbidden_overlap_loss(pred_prob, forbidden)
        area_pen = area_prior_loss(pred_prob, matched_gt, roi=roi)
        exist_ce, card = existence_losses(exist_logits, [m[0] for m in matches], K_gt)

        loss = (
            cfg.w_dice * dice_term
            + cfg.w_bce * bce_term
            + cfg.w_overlap * overlap
            + cfg.w_tv * tv
            + cfg.w_boundary * boundary
            + cfg.w_forbidden * forbid_pen
            + cfg.w_area * area_pen
            + cfg.w_exist_ce * exist_ce
            + cfg.w_cardinality * card
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: FullModel, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    matched_total = 0
    for batch in loader:
        image = batch["image"].to(cfg.device)
        aux = batch["aux"].to(cfg.device)
        gt_masks = batch["gt_masks"].to(cfg.device)
        gt_valid = batch["gt_valid"].to(cfg.device)

        roi = aux[:, cfg.roi_channel_index : cfg.roi_channel_index + 1]
        mask_logits, exist_logits, _ = model(image, aux)
        pred_prob = mask_logits.sigmoid()
        cost = dice_cost(mask_logits, gt_masks, roi=roi)
        matches = hungarian_match(cost, gt_valid)

        matched_gt = torch.zeros_like(mask_logits)
        matched_mask = torch.zeros((mask_logits.shape[0], mask_logits.shape[1]), dtype=torch.bool, device=mask_logits.device)
        for b, (rows, cols) in enumerate(matches):
            if rows.numel() == 0:
                continue
            matched_gt[b, rows] = gt_masks[b, cols]
            matched_mask[b, rows] = True

        dice_vals = 1.0 - dice_loss(pred_prob, matched_gt, mask=roi)
        if matched_mask.any():
            total_dice += float(dice_vals[matched_mask].mean().item())
            matched_total += 1
    if matched_total == 0:
        return {"dice": 0.0}
    return {"dice": total_dice / matched_total}


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
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data for smoke testing")
    parser.add_argument("--synthetic_samples", type=int, default=32, help="Number of synthetic samples")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
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

    for epoch in range(start_epoch, cfg.max_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
        elapsed = time.time() - start_time
        metrics = {"dice": 0.0}
        if val_loader is not None:
            metrics = evaluate(model, val_loader, cfg)
            if metrics["dice"] > best_metric:
                best_metric = metrics["dice"]
                save_checkpoint(save_dir / "best.pt", model, optimizer, epoch + 1, cfg, best_metric)
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch + 1, cfg, best_metric)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_dice": metrics.get("dice", 0.0),
                    "elapsed_sec": elapsed,
                }
            )
        )


if __name__ == "__main__":
    main()
