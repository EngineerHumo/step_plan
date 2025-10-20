"""Model components for the surgical planning set prediction network."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_unet import SpatialSelfAttention
from config import Config


class InputFusion(nn.Module):
    """Fuse raw image and auxiliary masks into a backbone-ready tensor."""

    def __init__(self, in_img_c: int, in_aux_c: int, out_c: int = 3) -> None:
        super().__init__()
        self.img_stem = nn.Sequential(
            nn.Conv2d(in_img_c, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.aux_stem = nn.Sequential(
            nn.Conv2d(in_aux_c, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, img: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_stem(img)
        aux_feat = self.aux_stem(aux)
        fused = torch.cat([img_feat, aux_feat], dim=1)
        return self.fuse(fused)


class DoubleConv(nn.Module):
    """Two consecutive Conv-InstanceNorm-LeakyReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """UNet encoder block with 2x down-sampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """UNet decoder block with transposed convolution up-sampling."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegFormerBackbone(nn.Module):
    """Six-stage UNet encoder that prepares features for the decoder."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_channels: List[int] = [32, 64, 128, 256, 512, 512]

        in_channels = 3
        self.stem = DoubleConv(in_channels, self.out_channels[0])
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(self.out_channels[i], self.out_channels[i + 1])
                for i in range(len(self.out_channels) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        cur = self.stem(x)
        feats.append(cur)
        for block in self.down_blocks:
            cur = block(cur)
            feats.append(cur)
        return feats


class PixelDecoder(nn.Module):
    """UNet decoder with spatial self-attention at the bottleneck."""

    def __init__(self, in_channels_list: List[int], embed_dim: int, attention_heads: int = 4) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("in_channels_list must not be empty")
        self.embed_dim = embed_dim
        self.attention = SpatialSelfAttention(embed_dim, num_heads=attention_heads)
        self.attention_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.up_blocks = nn.ModuleList(
            [UpBlock(embed_dim, embed_dim, embed_dim) for _ in range(len(in_channels_list) - 1)]
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if not feats:
            raise ValueError("PixelDecoder received no features")
        x = feats[-1]
        attended = self.attention(x)
        x = self.attention_fuse(torch.cat([x, attended], dim=1))

        skips = list(reversed(feats[:-1]))
        for idx, up in enumerate(self.up_blocks):
            skip = skips[idx]
            x = up(x, skip)
        return x


class MaskDecoder(nn.Module):
    """Heads for producing query masks, existence logits, and kernel embeddings."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg.num_queries
        E = cfg.mask_embed_dim
        D = cfg.embed_dim

        self.segmentation_head = nn.Conv2d(D, K, kernel_size=1, bias=True)
        self.exist_pool = nn.AdaptiveAvgPool2d(1)
        self.exist_fc = nn.Linear(D, K)
        self.kernel_proj = nn.Conv2d(D, K * E, kernel_size=1, bias=True)

    def forward(
        self, feat_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_logits = self.segmentation_head(feat_embed)
        exist_logits = self.exist_fc(self.exist_pool(feat_embed).flatten(1))

        kernel_map = self.kernel_proj(feat_embed)
        kernels = F.adaptive_avg_pool2d(kernel_map, 1).view(
            feat_embed.size(0), self.cfg.num_queries, self.cfg.mask_embed_dim
        )

        lowres_size = (
            max(1, mask_logits.shape[-2] // 4),
            max(1, mask_logits.shape[-1] // 4),
        )
        lowres_masks = (
            F.interpolate(
                mask_logits,
                size=lowres_size,
                mode="bilinear",
                align_corners=False,
            )
            if min(mask_logits.shape[-2:]) > 1
            else mask_logits
        )

        return mask_logits, exist_logits, lowres_masks, kernels
