from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Global configuration for model, training, and export."""

    img_size: Tuple[int, int] = (1024, 1024)
    in_channels: int = 3
    aux_mask_channels: int = 4
    num_queries: int = 6
    embed_dim: int = 256
    mask_embed_dim: int = 128
    mha_heads: int = 8
    feature_dims: Tuple[int, int, int, int] = (64, 128, 320, 512)
    use_timm: bool = True
    backbone_name: str = "mit_b0"
    roi_channel_index: int = 3

    # Loss weights
    w_dice: float = 1.15
    w_bce: float = 0.6
    w_overlap: float = 0.25
    w_tv: float = 0.05
    w_boundary: float = 0.2
    w_forbidden: float = 1.0
    w_area: float = 0.05
    w_exist_ce: float = 0.5
    w_cardinality: float = 0.2
    w_unmatched_dice: float = 0.5
    w_unmatched_bce: float = 0.25
    w_roi_background: float = 0.3
    w_matched_fp: float = 0.45
    w_unmatched_spill: float = 0.35
    w_query_cluster: float = 0.4
    w_query_compact: float = 0.3

    # Training parameters
    lr: float = 3e-4
    batch_size: int = 4
    max_epochs: int = 100
    device: str = "cuda"
    save_dir: str = "runs/exp0"
    val_interval: int = 10

    # Data augmentation parameters (inspired by nnU-Net defaults)
    aug_horizontal_flip_prob: float = 0.5
    aug_vertical_flip_prob: float = 0.5
    aug_rot90_prob: float = 0.5
    aug_scale_prob: float = 0.35
    aug_scale_range: Tuple[float, float] = (0.85, 1.2)
    aug_gamma_prob: float = 0.3
    aug_gamma_range: Tuple[float, float] = (0.7, 1.5)
    aug_brightness_prob: float = 0.25
    aug_brightness_std: float = 0.25
    aug_contrast_prob: float = 0.25
    aug_contrast_std: float = 0.3
    aug_gaussian_noise_prob: float = 0.2
    aug_gaussian_noise_std: float = 0.03
    aug_gaussian_blur_prob: float = 0.15
    aug_gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.2)

    # Visdom visualisation parameters
    visdom_enabled: bool = False
    visdom_env: str = "surgical-plan"
    visdom_server: str = "http://localhost"
    visdom_port: int = 8097
    visdom_max_samples: int = 4
