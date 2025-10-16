"""Model components for the surgical planning set prediction network."""

from __future__ import annotations

from typing import List

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class InputFusion(nn.Module):
    """Fuse raw image and auxiliary masks into a 3-channel tensor."""

    def __init__(self, in_img_c: int, in_aux_c: int, out_c: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_img_c + in_aux_c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_c, kernel_size=3, padding=1),
        )

    def forward(self, img: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img, aux], dim=1)
        return self.conv(x)


class SegFormerBackbone(nn.Module):
    """Wrapper around a SegFormer backbone with optional fallback."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder: nn.Module | None = None
        self.out_channels: List[int]
        if cfg.use_timm:
            try:
                import timm

                self.encoder = timm.create_model(
                    cfg.backbone_name,
                    features_only=True,
                    out_indices=(1, 2, 3, 4),
                    pretrained=True,
                )
                self.out_channels = list(self.encoder.feature_info.channels())
            except Exception:
                self.encoder = None
        if self.encoder is None:
            self.out_channels = list(cfg.feature_dims)
            layers: List[nn.Module] = []
            in_ch = 3
            for out_ch in cfg.feature_dims:
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
                layers.append(block)
                in_ch = out_ch
            self.fallback = nn.ModuleList(layers)
        else:
            self.fallback = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.encoder is not None:
            feats: List[torch.Tensor] = self.encoder(x)
            return feats
        feats: List[torch.Tensor] = []
        cur = x
        for block in self.fallback:
            cur = block(cur)
            feats.append(cur)
        return feats


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
        self.query_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.query_interaction = MHA(D, cfg.mha_heads)
        self.query_norm = nn.LayerNorm(D)
        self.query_ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.ReLU(inplace=True),
            nn.Linear(D * 4, D),
        )
        self.query_ffn_norm = nn.LayerNorm(D)

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
        queries = queries + self.query_interaction(queries, queries, queries)
        queries = self.query_norm(queries)
        queries = queries + self.query_ffn(queries)
        queries = self.query_ffn_norm(queries)
        queries = self.query_mlp(queries)
        queries = self.mha(queries, kv, kv)
        queries = self.norm(queries)

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
