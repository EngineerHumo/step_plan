from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Global configuration for model, training, and export."""

    img_size: Tuple[int, int] = (1024, 1024)
    in_channels: int = 1
    aux_mask_channels: int = 4
    num_queries: int = 4
    embed_dim: int = 256
    mask_embed_dim: int = 128
    mha_heads: int = 8
    feature_dims: Tuple[int, int, int, int] = (64, 128, 320, 512)
    use_timm: bool = True
    backbone_name: str = "mit_b0"
    roi_channel_index: int = 0

    # Loss weights
    w_dice: float = 1.0
    w_bce: float = 0.5
    w_overlap: float = 0.2
    w_tv: float = 0.05
    w_boundary: float = 0.2
    w_forbidden: float = 1.0
    w_area: float = 0.05
    w_exist_ce: float = 0.5
    w_cardinality: float = 0.2

    # Training parameters
    lr: float = 3e-4
    batch_size: int = 4
    max_epochs: int = 100
    device: str = "cuda"
    save_dir: str = "runs/exp0"
