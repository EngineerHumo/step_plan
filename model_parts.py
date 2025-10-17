"""Model components for the surgical planning set prediction network."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SegFormerBackbone(nn.Module):
    """Wrapper around a SegFormer backbone with optional fallback."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder: nn.Module | None = None
        self.out_channels: List[int]
        if cfg.use_pretrained_backbone:
            try:
                from transformers import SegformerModel

                self.encoder = SegformerModel.from_pretrained(cfg.backbone_name)
                self.out_channels = list(self.encoder.config.hidden_sizes)
                print(
                    "[SegFormerBackbone] Using Hugging Face SegFormer backbone "
                    f"'{cfg.backbone_name}'."
                )
            except Exception as exc:
                self.encoder = None
                print(
                    "[SegFormerBackbone] Failed to load pretrained SegFormer model "
                    f"'{cfg.backbone_name}': {exc}. Falling back to custom encoder."
                )
        if self.encoder is None:
            self.out_channels = list(cfg.feature_dims)
            if cfg.use_pretrained_backbone:
                print("[SegFormerBackbone] Initialising custom fallback encoder.")
            else:
                print(
                    "[SegFormerBackbone] Using custom encoder (pretrained backbone disabled in config)."
                )
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
            encoder_outputs = self.encoder(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = encoder_outputs.hidden_states
            if hidden_states is None or len(hidden_states) < 4:
                raise RuntimeError(
                    "[SegFormerBackbone] Expected hidden states from pretrained encoder."
                )
            selected_states = hidden_states[-4:]
            height, width = x.shape[-2], x.shape[-1]
            resolutions: List[tuple[int, int]] = []
            cur_h, cur_w = height // 4, width // 4
            resolutions.append((cur_h, cur_w))
            for _ in range(3):
                cur_h = max(cur_h // 2, 1)
                cur_w = max(cur_w // 2, 1)
                resolutions.append((cur_h, cur_w))
            feats: List[torch.Tensor] = []
            for state, (h, w) in zip(selected_states, resolutions):
                if state.dim() == 4:
                    feat = state
                elif state.dim() == 3:
                    B, seq_len, C = state.shape
                    if h * w != seq_len:
                        raise RuntimeError(
                            "[SegFormerBackbone] Hidden state sequence length does not "
                            "match expected spatial resolution."
                        )
                    feat = state.transpose(1, 2).reshape(B, C, h, w)
                else:
                    raise RuntimeError(
                        "[SegFormerBackbone] Unsupported hidden state dimensionality."
                    )
                feats.append(feat)
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
