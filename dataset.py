"""Dataset utilities for surgical sub-region planning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config


class SurgicalPlanningDataset(Dataset):
    """Dataset that loads images, auxiliary masks, and instance masks."""

    def __init__(self, items: Iterable[Dict[str, Any]], cfg: Config, augment: bool = True) -> None:
        self.items: List[Dict[str, Any]] = list(items)
        if not self.items:
            raise ValueError("Dataset received no items.")
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def _load_image_array(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        if path.suffix.lower() == ".npy":
            data = np.load(path).astype(np.float32)
            if data.ndim == 2:
                data = data[None]
            elif data.ndim == 3 and data.shape[0] not in (1, 3):
                data = data.transpose(2, 0, 1)
            return data
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = img[None]
        else:
            img = img.transpose(2, 0, 1)
        return img.astype(np.float32)

    def _read_image(self, path: str | Path) -> np.ndarray:
        img = self._load_image_array(path)
        if self.cfg.in_channels == 1 and img.shape[0] != 1:
            if img.shape[0] == 3:
                img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)[None]
            else:
                img = img[:1]
        elif self.cfg.in_channels == 3 and img.shape[0] != 3:
            if img.shape[0] == 1:
                img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)
            else:
                raise ValueError(f"Unsupported channel layout for image {path}")
        resized = cv2.resize(img.transpose(1, 2, 0), self.cfg.img_size[::-1], interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[None]
        else:
            resized = resized.transpose(2, 0, 1)
        mean = resized.mean()
        std = resized.std()
        return (resized - mean) / (std + 1e-6)

    def _read_aux_masks(self, paths: Iterable[str | Path]) -> np.ndarray:
        masks: List[np.ndarray] = []
        for path in paths:
            path = Path(path)
            if path.suffix.lower() == ".npy":
                arr = np.load(path).astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Aux mask {path} must be 2D")
                mask = arr
            else:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(path)
            mask = cv2.resize(mask, self.cfg.img_size[::-1], interpolation=cv2.INTER_NEAREST)
            masks.append((mask > 0.5).astype(np.float32))
        if len(masks) != self.cfg.aux_mask_channels:
            raise ValueError(f"Expected {self.cfg.aux_mask_channels} aux masks, got {len(masks)}")
        return np.stack(masks, axis=0)

    def _read_gt_instances(self, paths: Iterable[str | Path]) -> np.ndarray:
        masks: List[np.ndarray] = []
        for path in paths:
            path = Path(path)
            if path.suffix.lower() == ".npy":
                arr = np.load(path).astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"GT mask {path} must be 2D")
                mask = arr
            else:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(path)
            mask = cv2.resize(mask, self.cfg.img_size[::-1], interpolation=cv2.INTER_NEAREST)
            masks.append((mask > 0.5).astype(np.float32))
        if masks:
            return np.stack(masks, axis=0)
        return np.zeros((0, *self.cfg.img_size), dtype=np.float32)

    def _augment(self, image: np.ndarray, aux: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=-1).copy()
            aux = np.flip(aux, axis=-1).copy()
            if gt.shape[0] > 0:
                gt = np.flip(gt, axis=-1).copy()
        return image, aux, gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        image = self._read_image(item["image"])
        aux = self._read_aux_masks(item["aux_masks"])
        gt = self._read_gt_instances(item.get("gt_masks", []))
        if self.augment:
            image, aux, gt = self._augment(image, aux, gt)
        roi = aux[self.cfg.roi_channel_index : self.cfg.roi_channel_index + 1]
        if gt.shape[0] > 0:
            gt = gt * roi
        return {
            "image": torch.from_numpy(image),
            "aux": torch.from_numpy(aux),
            "gt_masks": torch.from_numpy(gt),
            "K_gt": torch.tensor(gt.shape[0], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, Any]], cfg: Config) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    aux = torch.stack([b["aux"] for b in batch], dim=0)
    B = images.shape[0]
    H, W = cfg.img_size
    Kmax = cfg.num_queries
    gt_masks = torch.zeros((B, Kmax, H, W), dtype=torch.float32)
    gt_valid = torch.zeros((B, Kmax), dtype=torch.bool)
    K_gt_list: List[int] = []
    for i, b in enumerate(batch):
        total = int(b["K_gt"].item())
        k = min(total, Kmax)
        if k > 0:
            gt_masks[i, :k] = b["gt_masks"][:k]
            gt_valid[i, :k] = True
        K_gt_list.append(total)
    return {
        "image": images,
        "aux": aux,
        "gt_masks": gt_masks,
        "gt_valid": gt_valid,
        "K_gt": torch.tensor(K_gt_list, dtype=torch.long),
    }


def load_items(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset file must contain a list of items.")
    return data


def build_synthetic_items(root: Path, num_samples: int, cfg: Config) -> List[Dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    H, W = cfg.img_size
    for idx in range(num_samples):
        image = np.random.randn(cfg.in_channels, H, W).astype(np.float32)
        aux = np.zeros((cfg.aux_mask_channels, H, W), dtype=np.uint8)
        roi = (np.random.rand(H, W) > 0.2).astype(np.uint8)
        aux[0] = roi
        aux[1] = ((np.random.rand(H, W) > 0.95) & (roi == 1)).astype(np.uint8)
        aux[2] = ((np.random.rand(H, W) > 0.97) & (roi == 1)).astype(np.uint8)
        aux[3] = 1 - aux[0]
        num_inst = random.randint(1, cfg.num_queries)
        gt_paths: List[str] = []
        item_dir = root / f"sample_{idx:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        image_path = item_dir / "image.png"
        if cfg.in_channels == 1:
            norm = (image[0] - image[0].min()) / (image[0].ptp() + 1e-6)
            cv2.imwrite(str(image_path), (norm * 255).astype(np.uint8))
        else:
            norm = (image.transpose(1, 2, 0) - image.min()) / (image.ptp() + 1e-6)
            cv2.imwrite(str(image_path), (norm * 255).astype(np.uint8))
        aux_paths: List[str] = []
        for ch in range(cfg.aux_mask_channels):
            mask_path = item_dir / f"aux_{ch}.png"
            cv2.imwrite(str(mask_path), aux[ch] * 255)
            aux_paths.append(str(mask_path))
        for inst_idx in range(num_inst):
            center_x = random.randint(W // 4, 3 * W // 4)
            center_y = random.randint(H // 4, 3 * H // 4)
            radius = random.randint(min(H, W) // 20, min(H, W) // 8)
            Y, X = np.ogrid[:H, :W]
            mask = (((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2).astype(np.uint8)
            mask = mask * roi
            mask_path = item_dir / f"gt_{inst_idx}.png"
            cv2.imwrite(str(mask_path), mask * 255)
            gt_paths.append(str(mask_path))
        items.append(
            {
                "image": str(image_path),
                "aux_masks": aux_paths,
                "gt_masks": gt_paths,
            }
        )
    return items
