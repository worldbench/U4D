import argparse
from pathlib import Path
from typing import Tuple

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from utils import inference, render


def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    ddpm, lidar_utils, cfg = inference.setup_model(
        args.ckpt, device=args.device)

    if cfg.model.with_condition:
        from torch.utils.data import DataLoader

        from data.nuscenes import NuScenesDataset

        def cond_preprocess(batch: dict) -> torch.Tensor:
            x = []
            if cfg.data.train_depth:
                x += [lidar_utils.convert_depth(batch['cond_depth'])]
            if cfg.data.train_reflectance:
                x += [batch['cond_reflectance']]
            x = torch.cat(x, dim=2)
            x = lidar_utils.normalize(x).to(args.device)
            return x

        dataset = NuScenesDataset(
            data_root=cfg.data.data_root,
            uncertainty_root=cfg.data.uncertainty_root,
            rate=cfg.data.rate,
            split='val',
            resolution=cfg.data.resolution,
            num_frames=cfg.data.num_frames)

        cond_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            drop_last=False)
        cond_iter = iter(cond_dataloader)
        batch = next(cond_iter)
        cond = cond_preprocess(batch)
    else:
        cond = None

    xs = ddpm.sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        cond=cond,
        return_all=True).clamp(-1, 1)

    xs = lidar_utils.denormalize(xs)
    xs[:, :, :, [0]] = lidar_utils.revert_depth(
        xs[:, :, :, [0]]) / lidar_utils.max_depth

    def rendering(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 5:
            x = einops.rearrange(x, 'B T C H W -> (B T) C H W')
        img = einops.rearrange(x, 'B C H W -> B 1 (C H) W')
        img = render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
        xyz /= lidar_utils.max_depth
        z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
        z = (xyz[:, [2]] - z_min) / (z_max - z_min)
        colors = render.colorize(z.clamp(0, 1), cm.viridis) / 255
        R, t = render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
        bev = 1 - render.render_point_clouds(
            points=einops.rearrange(xyz, 'B C H W -> B (H W) C'),
            colors=1 - einops.rearrange(colors, 'B C H W -> B (H W) C'),
            R=R.to(xyz),
            t=t.to(xyz))
        return img, bev

    img, bev = rendering(xs[-1])
    save_image(img, 'samples_img.png', nrow=1)
    save_image(bev, 'samples_bev.png', nrow=4)

    video = imageio.get_writer('denoise.mp4', mode='I', fps=60)
    for x in tqdm(xs, desc='making video...'):
        img, bev = rendering(x)
        scale = 512 / img.shape[-1]
        img = F.interpolate(
            img, scale_factor=scale, mode='bilinear', antialias=True)
        scale = 512 / bev.shape[-1]
        bev = F.interpolate(
            bev, scale_factor=scale, mode='bilinear', antialias=True)
        img = torch.cat([img, bev], dim=2)
        img = make_grid(
            img, nrow=args.batch_size * cfg.data.num_frames, pad_value=1)
        img = img.permute(1, 2, 0).mul(255).byte()
        video.append_data(img.cpu().numpy())

    video = imageio.get_writer('sequences.mp4', mode='I', fps=2)
    sequence = xs[-1]
    sequence = einops.rearrange(sequence, 'B T C H W -> T B C H W')
    for x in tqdm(sequence, desc='making video...'):
        img, bev = rendering(x)
        scale = 512 / img.shape[-1]
        img = F.interpolate(
            img, scale_factor=scale, mode='bilinear', antialias=True)
        scale = 512 / bev.shape[-1]
        bev = F.interpolate(
            bev, scale_factor=scale, mode='bilinear', antialias=True)
        img = torch.cat([img, bev], dim=2)
        img = make_grid(img, nrow=args.batch_size, pad_value=1)
        img = img.permute(1, 2, 0).mul(255).byte()
        video.append_data(img.cpu().numpy())
    video.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sampling_steps', type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
