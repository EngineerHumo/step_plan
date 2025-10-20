from contextlib import nullcontext
from math import prod
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import StackedConvBlocks


class SpatialSelfAttention(nn.Module):
    """Applies multi-head self-attention across spatial positions of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 4096):
        super().__init__()
        num_heads = max(1, num_heads)
        head_dim = channels // num_heads
        if head_dim * num_heads != channels:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads}) for attention export"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(channels, channels * 3, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=True)

        self.norm = nn.LayerNorm(channels)
        self.max_tokens = max(1, int(max_tokens))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projections following nn.MultiheadAttention defaults."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _reduced_shape(spatial_dims: Sequence[int], max_tokens: int) -> Tuple[int, ...]:
        target = list(spatial_dims)
        while target and prod(target) > max_tokens:
            largest_axis = max(range(len(target)), key=lambda i: target[i])
            target[largest_axis] = max(1, (target[largest_axis] + 1) // 2)
        return tuple(target)

    @staticmethod
    def _adaptive_pool(x: torch.Tensor, output_size: Tuple[int, ...]) -> torch.Tensor:
        if len(output_size) == 1:
            return F.adaptive_avg_pool1d(x, output_size[0])
        if len(output_size) == 2:
            return F.adaptive_avg_pool2d(x, output_size)
        if len(output_size) == 3:
            return F.adaptive_avg_pool3d(x, output_size)
        raise ValueError(f"Unsupported tensor rank for adaptive pooling: {x.ndim}")

    @staticmethod
    def _interpolate(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        if len(size) == 1:
            return F.interpolate(x, size=size, mode="linear", align_corners=False)
        if len(size) == 2:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if len(size) == 3:
            return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        raise ValueError(f"Unsupported interpolation size: {size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            return x

        spatial_dims: Sequence[int] = x.shape[2:]
        orig_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"

        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if torch.is_autocast_enabled()
            else nullcontext()
        )

        with autocast_context:
            work_x = x if x.dtype == torch.float32 else x.to(torch.float32)
            identity = work_x
            target_dims = self._reduced_shape(spatial_dims, self.max_tokens)

            pooled = (
                self._adaptive_pool(work_x, target_dims)
                if target_dims != tuple(spatial_dims)
                else work_x
            )

            b, c = pooled.shape[:2]
            flattened = pooled.view(b, c, -1).permute(0, 2, 1)
            qkv = self.qkv_proj(flattened)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
                bsz, seq_len, dim = tensor.shape
                tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
                return tensor.permute(0, 2, 1, 3)

            q = reshape_heads(q).to(torch.float32)
            k = reshape_heads(k).to(torch.float32)
            v = reshape_heads(v).to(torch.float32)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)
            attn_weights = attn_scores.softmax(dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights, v)

            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(flattened.shape[0], flattened.shape[1], -1)

            attended = self.out_proj(attn_output)
            attended = attended.permute(0, 2, 1).contiguous().view(b, c, *target_dims)

            if target_dims != tuple(spatial_dims):
                attended = self._interpolate(attended, spatial_dims)

            attended = (attended + identity).contiguous()
            attended = attended.view(b, c, -1).permute(0, 2, 1)
            attended = self.norm(attended)
            attended = attended.permute(0, 2, 1).contiguous().view(b, c, *spatial_dims)

        if attended.dtype != orig_dtype:
            attended = attended.to(orig_dtype)

        attended = (attended + identity).contiguous()
        attended = attended.view(b, c, -1).permute(0, 2, 1)
        attended = self.norm(attended)
        attended = attended.permute(0, 2, 1).contiguous().view(b, c, *spatial_dims)

        if attended.dtype != orig_dtype:
            attended = attended.to(orig_dtype)

        return attended


class AttentionUNet(PlainConvUNet):
    """PlainConvUNet variant with a spatial self-attention bottleneck."""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op,
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, type] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, type] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, type] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        attention_heads: int = 4,
        attention_max_tokens: int = 4096,
    ):
        super().__init__(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            num_classes,
            n_conv_per_stage_decoder,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            deep_supervision,
            nonlin_first,
        )

        bottleneck_channels = self.encoder.output_channels[-1]
        self.attention_block = SpatialSelfAttention(
            bottleneck_channels,
            attention_heads,
            max_tokens=attention_max_tokens,
        )
        self.attention_fuse = StackedConvBlocks(
            num_convs=1,
            conv_op=self.encoder.conv_op,
            input_channels=bottleneck_channels * 2,
            output_channels=bottleneck_channels,
            kernel_size=self.encoder.kernel_sizes[-1],
            initial_stride=1,
            conv_bias=self.encoder.conv_bias,
            norm_op=self.encoder.norm_op,
            norm_op_kwargs=self.encoder.norm_op_kwargs,
            dropout_op=self.encoder.dropout_op,
            dropout_op_kwargs=self.encoder.dropout_op_kwargs,
            nonlin=self.encoder.nonlin,
            nonlin_kwargs=self.encoder.nonlin_kwargs,
            nonlin_first=nonlin_first,
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        bottleneck = skips[-1]
        attended = self.attention_block(bottleneck)
        fused = torch.cat((bottleneck, attended), dim=1)
        skips[-1] = self.attention_fuse(fused)
        return self.decoder(skips)
