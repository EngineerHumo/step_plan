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


def _conv_norm_leaky(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
        nn.LeakyReLU(inplace=True),
    )


class EncoderStage(nn.Module):
    """Stacked convolutions with optional strided down-sampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv_norm_leaky(in_channels, out_channels, stride=stride),
            _conv_norm_leaky(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegFormerBackbone(nn.Module):
    """Six-stage UNet encoder with a spatial self-attention bottleneck."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_channels: List[int] = [32, 64, 128, 256, 512, 512]
        strides = [1, 2, 2, 2, 2, 2]
        stages: List[nn.Module] = []
        in_ch = 3
        for out_ch, stride in zip(self.out_channels, strides):
            stages.append(EncoderStage(in_ch, out_ch, stride))
            in_ch = out_ch
        self.encoder = nn.ModuleList(stages)
        bottleneck_channels = self.out_channels[-1]
        self.attention = SpatialSelfAttention(
            bottleneck_channels,
            num_heads=4,
            max_tokens=4096,
        )
        self.attention_fuse = nn.Sequential(
            _conv_norm_leaky(bottleneck_channels * 2, bottleneck_channels),
            _conv_norm_leaky(bottleneck_channels, bottleneck_channels),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        cur = x
        for stage in self.encoder:
            cur = stage(cur)
            features.append(cur)
        bottleneck = features[-1]
        attended = self.attention(bottleneck)
        fused = torch.cat([bottleneck, attended], dim=1)
        features[-1] = self.attention_fuse(fused)
        return features


class PixelDecoder(nn.Module):
    """UNet-style decoder that produces spatial embeddings."""

    def __init__(self, in_channels_list: List[int], embed_dim: int) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("PixelDecoder requires at least one feature map")
        self.embed_dim = embed_dim
        projections: List[nn.Module] = []
        for ch in in_channels_list:
            if ch == embed_dim:
                projections.append(nn.Identity())
            else:
                projections.append(nn.Conv2d(ch, embed_dim, kernel_size=1, bias=True))
        self.projections = nn.ModuleList(projections)
        self.bottleneck = nn.Sequential(
            _conv_norm_leaky(embed_dim, embed_dim),
            _conv_norm_leaky(embed_dim, embed_dim),
        )
        self.decode_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            block = nn.Sequential(
                _conv_norm_leaky(embed_dim * 2, embed_dim),
                _conv_norm_leaky(embed_dim, embed_dim),
            )
            self.decode_blocks.append(block)
        self.out = nn.Sequential(
            _conv_norm_leaky(embed_dim, embed_dim),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.projections):
            raise ValueError("Mismatch between features and decoder projections")
        processed = [proj(feat) for proj, feat in zip(self.projections, feats)]
        x = self.bottleneck(processed[-1])
        for idx in range(len(processed) - 2, -1, -1):
            skip = processed[idx]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            block_index = len(processed) - 2 - idx
            x = self.decode_blocks[block_index](x)
        return self.out(x)


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
