"""Training script for permutation-invariant surgical sub-region segmentation."""

from __future__ import annotations

import argparse
import json
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
    query_diversity_loss,
    query_kernel_diversity_loss,
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

    def _to_grid(self, tensor: torch.Tensor, normalize: bool = True) -> np.ndarray | None:
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
            nrow=max(1, min(data.size(0), self.cfg.visdom_max_samples)),
            padding=2,
        )
        return grid.cpu().numpy()

    def _show(self, key: str, img: np.ndarray | None, title: str) -> None:
        if not self.enabled or self.vis is None or img is None:
            return
        opts = {"title": f"{title} (step {self.global_step})"}
        win = self._wins.get(key)
        self._wins[key] = self.vis.image(img, win=win, opts=opts)
    

    def log_batch(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_prob: torch.Tensor,
    ) -> None:
        if not self.enabled or self.vis is None:
            return
        try:
            # 显示输入图像
            inputs_grid = self._to_grid(images, normalize=True)
        
            # 为每个通道创建单独的显示
            B, C, H, W = gt_masks.shape
        
            # 显示每个通道的GT masks
            for c in range(C):
                gt_channel = gt_masks[:, c:c+1]  # [B, 1, H, W]
                gt_grid = self._to_grid(gt_channel, normalize=True)
                self._show(f"labels_channel_{c}", gt_grid, f"Train/Label_Channel_{c}")
        
            # 显示每个通道的预测概率
            for c in range(C):
                pred_channel = pred_prob[:, c:c+1]  # [B, 1, H, W]
                pred_grid = self._to_grid(pred_channel, normalize=True)
                self._show(f"outputs_channel_{c}", pred_grid, f"Train/Output_Channel_{c}")
        
            # 可选：显示合并视图（所有通道的最大值）
            gt_combined = gt_masks.float().max(dim=1).values.unsqueeze(1)
            gt_grid = self._to_grid(gt_combined, normalize=True)
        
            pred_combined = pred_prob.max(dim=1).values.unsqueeze(1)
            pred_grid = self._to_grid(pred_combined, normalize=True)
        
            self._show("inputs", inputs_grid, "Train/Input")
            self._show("labels_combined", gt_grid, "Train/Label_Combined")
            self._show("outputs_combined", pred_grid, "Train/Output_Combined")
        
            self.global_step += 1
        except Exception as exc:
            print(f"[Visdom] Logging error: {exc}")
            self.enabled = False
            self.vis = None

    '''
    def log_batch(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_prob: torch.Tensor,
    ) -> None:
        if not self.enabled or self.vis is None:
            return
        try:
            inputs_grid = self._to_grid(images, normalize=True)
            gt_combined = gt_masks.float().max(dim=1).values.unsqueeze(1)
            gt_grid = self._to_grid(gt_combined, normalize=True)
            pred_combined = pred_prob.max(dim=1).values.unsqueeze(1)
            pred_grid = self._to_grid(pred_combined, normalize=True)
            self._show("inputs", inputs_grid, "Train/Input")
            self._show("labels", gt_grid, "Train/Label")
            self._show("outputs", pred_grid, "Train/Output")
            self.global_step += 1
        except Exception as exc:  # pragma: no cover - visual only
            print(f"[Visdom] Logging error: {exc}")
            self.enabled = False
            self.vis = None
    '''


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
        return torch.ones(shape, dtype=aux.dtype, device=aux.device)

    roi = aux[:, cfg.roi_channel_index : cfg.roi_channel_index + 1].clone()
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

        roi = _get_roi_mask(aux, cfg)
        forbidden = aux[:, 1:3].sum(1, keepdim=True).clamp(max=1.0)

        mask_logits, exist_logits, decoder_out = model(image, aux)
        pred_prob = mask_logits.sigmoid()

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

        diversity_loss = query_diversity_loss(pred_prob, mask=roi)
        kernel_div = query_kernel_diversity_loss(decoder_out["kernels"])

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
            + 0.1 * diversity_loss
            + 0.05 * kernel_div
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        print(gt_masks.max())
        print(pred_prob.max())
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
            vis_logger.log_batch(image, gt_masks, pred_prob)
            #vis_logger.log_batch(image, gt_masks[:,0,:,:], pred_prob[:,0,:,:],gt_masks[:,1,:,:], pred_prob[:,0,:,:],gt_masks[:,0,:,:], pred_prob[:,0,:,:],gt_masks[:,0,:,:], pred_prob[:,0,:,:])
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

        roi = _get_roi_mask(aux, cfg)
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
