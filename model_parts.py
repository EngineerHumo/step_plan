"""Model components for the surgical planning set prediction network."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

from attention_unet import SpatialSelfAttention


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


class ConvBlock(nn.Module):
    """Two stacked convolutions with InstanceNorm and LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample by transposed convolution and fuse with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # pragma: no cover - guard for shape mismatches
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegFormerBackbone(nn.Module):
    """Static 6-stage U-Net encoder-decoder with spatial self-attention bottleneck."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Lighter channel configuration to reduce peak feature map memory
        self.features_per_stage: List[int] = [24, 48, 96, 192, 256, 256]
        self.stage_strides: List[int] = [1, 2, 2, 2, 2, 2]
        self.encoder_stages = nn.ModuleList()

        in_ch = 3
        for out_ch, stride in zip(self.features_per_stage, self.stage_strides):
            self.encoder_stages.append(ConvBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        heads = cfg.mha_heads if self.features_per_stage[-1] % cfg.mha_heads == 0 else 4
        self.attention = SpatialSelfAttention(
            self.features_per_stage[-1],
            num_heads=max(1, heads),
            max_tokens=4096,
        )
        self.attention_fuse = ConvBlock(
            self.features_per_stage[-1] * 2,
            self.features_per_stage[-1],
            stride=1,
        )

        # Only decode up to 1/4 resolution to keep downstream tensors compact
        decoder_specs = [
            (self.features_per_stage[-1], self.features_per_stage[-2], self.features_per_stage[-2]),
            (self.features_per_stage[-2], self.features_per_stage[-3], self.features_per_stage[-3]),
            (self.features_per_stage[-3], self.features_per_stage[-4], self.features_per_stage[-4]),
        ]
        self.decoder_stages = nn.ModuleList(
            [UpBlock(in_ch, skip_ch, out_ch) for in_ch, skip_ch, out_ch in decoder_specs]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.features_per_stage[2], cfg.embed_dim, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(cfg.embed_dim, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(cfg.embed_dim, cfg.embed_dim, kernel_size=1, bias=True),
        )
        # Decoder exposes multi-scale features ordered from high to low resolution
        decoder_out_channels = [spec[-1] for spec in decoder_specs][::-1]
        self.out_channels: List[int] = [cfg.embed_dim] + decoder_out_channels[1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = []
        cur = x
        for stage in self.encoder_stages:
            cur = stage(cur)
            features.append(cur)

        bottleneck = features[-1]
        attended = self.attention(bottleneck)
        fused = torch.cat([bottleneck, attended], dim=1)
        cur = self.attention_fuse(fused)

        decoder_feats: List[torch.Tensor] = []
        skips = features[:-1][::-1]
        for stage, skip in zip(self.decoder_stages, skips):
            cur = stage(cur, skip)
            decoder_feats.append(cur)

        high_to_low = decoder_feats[::-1]
        final_feature = self.final_conv(high_to_low[0])
        return [final_feature] + high_to_low[1:]


class PixelDecoder(nn.Module):
    """Top-down fusion decoder that produces per-pixel embeddings."""

    def __init__(self, in_channels_list: List[int], embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list])
        self.fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x: torch.Tensor | None = None
        for idx, feat in enumerate(reversed(feats)):
            proj = self.proj[len(feats) - 1 - idx](feat)
            if x is None:
                x = proj
            else:
                x = F.interpolate(x, size=proj.shape[-2:], mode="bilinear", align_corners=False) + proj
        assert x is not None
        x = self.fuse(x)
        return x


class SimplePositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe(x)


class MHA(nn.Module):
    """Manual multi-head attention implementation for ONNX export."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        q_proj = self.q_proj(q).view(B, Nq, self.nhead, self.dk).transpose(1, 2)
        k_proj = self.k_proj(k).view(B, Nk, self.nhead, self.dk).transpose(1, 2)
        v_proj = self.v_proj(v).view(B, Nk, self.nhead, self.dk).transpose(1, 2)
        attn = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / (self.dk ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        return self.out(out)


class MaskDecoder(nn.Module):
    """Query-mask decoder producing low-resolution masks and existence logits."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        E = cfg.mask_embed_dim
        K = cfg.num_queries

        self.query_embed = nn.Parameter(torch.randn(K, D) * 0.1)

        with torch.no_grad():
            pos_encoding = torch.zeros(K, D)
            for i in range(K):
                for j in range(0, D, 2):
                    pos_encoding[i, j] = math.sin(i / (10000 ** (2 * j / D)))
                    if j + 1 < D:
                        pos_encoding[i, j + 1] = math.cos(i / (10000 ** (2 * j / D)))
            self.query_embed.data += pos_encoding * 0.1
        self.query_norm = nn.LayerNorm(D)
        self.query_interaction = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.query_dropout = nn.Dropout(cfg.query_dropout if hasattr(cfg, "query_dropout") else 0.1)

        self.mha = MHA(D, cfg.mha_heads)
        self.norm = nn.LayerNorm(D)

        self.pixel_proj = nn.Conv2d(D, E, kernel_size=1)
        self.posenc = SimplePositionalEncoding(D)

        self.kernel_head = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, E),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, 1),
        )


    def forward(
        self, feat_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, H4, W4 = feat_embed.shape
        pos = self.posenc(feat_embed)
        kv = pos.flatten(2).transpose(1, 2)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        base_queries = self.query_norm(queries)
        queries = queries + self.query_dropout(self.query_interaction(base_queries))
        queries = queries + self.query_dropout(self.query_mlp(self.query_norm(queries)))
        attn_out = self.mha(self.query_norm(queries), kv, kv)
        queries = self.norm(queries + attn_out)

        kernels = self.kernel_head(queries)
        exist_logits = self.exist_head(queries).squeeze(-1)

        pix_embed = self.pixel_proj(feat_embed)
        lowres_masks = torch.einsum("bke,behw->bkhw", kernels, pix_embed)
        mask_logits = F.interpolate(
            lowres_masks,
            size=self.cfg.img_size,
            mode="bilinear",
            align_corners=False,
        )
        return mask_logits, exist_logits, lowres_masks, kernels
