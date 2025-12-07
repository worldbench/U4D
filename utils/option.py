from typing import Optional, Tuple

from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    base_channels: int = 64
    temb_channels: Optional[int] = None
    channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8)
    num_residual_blocks: Tuple[int, ...] = (3, 3, 3, 3)
    gn_num_groups: int = 32 // 4
    gn_eps: float = 1e-6
    attn_num_heads: int = 8
    coords_encoding: str = 'fourier_features'
    with_condition: bool = False


@dataclass
class DiffusionConfig:
    num_sampling_steps: int = 1024
    prediction_type: str = 'eps'
    loss_type: str = 'l2'
    noise_schedule: str = 'cosine'


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    num_steps: int = 500_000
    steps_save_image: int = 5_000
    steps_save_model: int = 500_000
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    lr_warmup_steps: int = 10_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    mixed_precision: str = 'fp16'
    dynamo_backend: str = 'inductor'
    output_dir: str = 'logs/diffusion'


@dataclass
class DataConfig:
    data_root: str = '/home/ac/data/nuscenes'
    uncertainty_root: str = '/home/ac/data/nuscenes_entropy'
    rate: float = 0.3
    depth_format: str = 'log_depth'
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (32, 1024)
    min_depth: float = 1.45
    max_depth: float = 80.0
    num_frames: int = 6


@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
