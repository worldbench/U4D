import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path
from typing import Tuple

import einops
import matplotlib.cm as cm
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from rich import print
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.nuscenes import NuScenesDataset
from models.diffusion import GaussianDiffusion
from models.efficient_unet import EfficientUNet
from utils import inference, lidar, option, render, training

warnings.filterwarnings('ignore', category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


def train(cfg: option.Config) -> None:
    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.training.output_dir)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=['tensorboard'],
        project_dir=project_dir,
        dynamo_backend=cfg.training.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True)
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker('tensorboard')
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / 'training_config.json', 'w'),
            indent=4)
    device = accelerator.device

    # Setup models
    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
    ]

    model = EfficientUNet(
        in_channels=sum(channels),
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

    model.coords = lidar.get_hdl_linear_ray_angles(*cfg.data.resolution)

    if accelerator.is_main_process:
        print(f'number of parameters: {inference.count_parameters(model):,}')

    ddpm = GaussianDiffusion(
        model=model,
        prediction_type=cfg.diffusion.prediction_type,
        loss_type=cfg.diffusion.loss_type,
        noise_schedule=cfg.diffusion.noise_schedule)
    ddpm.train()
    ddpm.to(device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=cfg.training.lr_warmup_steps *
            cfg.training.gradient_accumulation_steps)
        ddpm_ema.to(device)

    lidar_utils = lidar.LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords)
    lidar_utils.to(device)

    # Setup optimizer & dataloader
    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon)

    dataset = NuScenesDataset(
        data_root=cfg.data.data_root,
        uncertainty_root=cfg.data.uncertainty_root,
        rate=cfg.data.rate,
        split='train',
        resolution=cfg.data.resolution,
        num_frames=cfg.data.num_frames)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True)

    lr_scheduler = training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps *
        cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.num_steps *
        cfg.training.gradient_accumulation_steps)

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler)

    # Utility

    def preprocess(batch: dict) -> torch.Tensor:
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch['depth'])]
        if cfg.data.train_reflectance:
            x += [batch['reflectance']]
        x = torch.cat(x, dim=2)
        x = lidar_utils.normalize(x).to(device)
        return x

    def cond_preprocess(batch: dict) -> torch.Tensor:
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch['cond_depth'])]
        if cfg.data.train_reflectance:
            x += [batch['cond_reflectance']]
        x = torch.cat(x, dim=2)
        x = lidar_utils.normalize(x).to(device)
        return x

    def split_channels(
            image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image: torch.Tensor,
                   tag: str = 'name',
                   global_step: int = 0) -> None:
        image = einops.rearrange(image, 'B T C H W -> (B T) C H W')
        image = lidar_utils.denormalize(image)
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f'{tag}/depth'] = render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (
                metric < lidar_utils.max_depth)
            out[f'{tag}/depth/orig'] = render.colorize(metric /
                                                       lidar_utils.max_depth)
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            bev = render.render_point_clouds(
                points=einops.rearrange(xyz, 'B C H W -> B (H W) C'),
                colors=einops.rearrange(normal, 'B C H W -> B (H W) C'),
                t=torch.tensor([0, 0, 1.0]).to(xyz))
            out[f'{tag}/bev'] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f'{tag}/reflectance'] = render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f'{tag}/mask'] = render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    progress_bar = tqdm(
        range(cfg.training.num_steps),
        desc='training',
        dynamic_ncols=True,
        disable=not accelerator.is_main_process)

    global_step = 0
    while global_step < cfg.training.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_0 = preprocess(batch)
            if cfg.model.with_condition:
                cond = cond_preprocess(batch)
            else:
                cond = None

            with accelerator.accumulate(ddpm):
                loss = ddpm(x_0=x_0, cond=cond)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log['ema_decay'] = ddpm_ema.get_current_decay()

                if global_step == 1:
                    log_images(x_0, 'images', global_step)
                    if cfg.model.with_condition:
                        log_images(cond, 'image_cond', global_step)

                if global_step % cfg.training.steps_save_image == 0:
                    ddpm_ema.ema_model.eval()
                    sample = ddpm_ema.ema_model.sample(
                        batch_size=cfg.training.batch_size //
                        accelerator.num_processes,
                        num_steps=cfg.diffusion.num_sampling_steps,
                        cond=cond,
                        rng=torch.Generator(device='cuda').manual_seed(0))
                    log_images(sample, 'sample', global_step)
                    if cfg.model.with_condition:
                        log_images(cond, 'sample_cond', global_step)

                if global_step % cfg.training.steps_save_model == 0:
                    save_dir = Path(tracker.logging_dir) / 'models'
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            'cfg': dataclasses.asdict(cfg),
                            'weights': ddpm_ema.online_model.state_dict(),
                            'ema_weights': ddpm_ema.ema_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'global_step': global_step,
                        }, save_dir / f'diffusion_{global_step:010d}.pth')

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.training.num_steps:
                break

    accelerator.end_training()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(option.Config, dest='cfg')
    train(parser.parse_args().cfg)
