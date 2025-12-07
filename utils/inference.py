from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import nn

from models.diffusion import GaussianDiffusion
from models.efficient_unet import EfficientUNet
from .lidar import LiDARUtility
from .option import Config


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_model(
        ckpt: Union[str, Path, dict],
        device: Union[torch.device, str] = 'cpu',
        ema: bool = True,
        show_info: bool = True,
        compile: bool = False
) -> Tuple[GaussianDiffusion, LiDARUtility, Config]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location='cpu')
    cfg = Config(**ckpt['cfg'])

    in_channels = [0, 0]
    if cfg.data.train_depth:
        in_channels[0] = 1
    if cfg.data.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    model = EfficientUNet(
        in_channels=in_channels,
        resolution=cfg.data.resolution,
        base_channels=cfg.model.base_channels,
        temb_channels=cfg.model.temb_channels,
        channel_multiplier=cfg.model.channel_multiplier,
        num_residual_blocks=cfg.model.num_residual_blocks,
        gn_num_groups=cfg.model.gn_num_groups,
        gn_eps=cfg.model.gn_eps,
        attn_num_heads=cfg.model.attn_num_heads,
        coords_encoding=cfg.model.coords_encoding,
        num_frames=cfg.data.num_frames,
        with_condition=cfg.model.with_condition,
        ring=True)

    ddpm = GaussianDiffusion(
        model=model,
        prediction_type=cfg.diffusion.prediction_type,
        loss_type=cfg.diffusion.loss_type,
        noise_schedule=cfg.diffusion.noise_schedule)
    state_dict = ckpt['ema_weights'] if ema else ckpt['weights']
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)

    if compile:
        ddpm.model = torch.compile(ddpm.model)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords)
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f'resolution: {model.resolution}',
                f'model: {model.__class__.__name__}',
                f'ddpm: {ddpm.__class__.__name__}',
                f'#steps: {ckpt["global_step"]:,}',
                f'#params: {count_parameters(ddpm):,}',
            ],
            sep='\n')
    return ddpm, lidar_utils, cfg


def setup_rng(seeds: List[int], device: Union[torch.device,
                                              str]) -> List[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
