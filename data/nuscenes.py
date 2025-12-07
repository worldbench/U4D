import os
import pickle
from typing import Tuple

import numba
import numpy as np
from torch.utils.data import Dataset


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(point_path: str,
                          H: int = 32,
                          W: int = 2048,
                          min_depth: float = 1.45,
                          max_depth: float = 80.0) -> np.ndarray:
    # load xyz & intensity and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 5)[:, :4]
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth < max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    # vertical grid
    h_up, h_down = np.deg2rad(10), np.deg2rad(-30)
    elevation = np.arcsin(z / depth) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    azimuth = -np.arctan2(y, x)
    grid_w = (azimuth / np.pi + 1) / 2 % 1
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)


class NuScenesDataset(Dataset):

    def __init__(self,
                 data_root: str,
                 uncertainty_root: str,
                 rate: float = 0.3,
                 split: str = 'train',
                 resolution: Tuple[int, int] = (32, 1024),
                 num_frames: int = 6) -> None:
        super(NuScenesDataset, self).__init__()
        self.data_root = data_root
        self.uncertainty_root = uncertainty_root
        self.resolution = resolution
        self.num_frames = num_frames
        self.num_uncertainty = int(self.resolution[0] * self.resolution[1] *
                                   rate)

        if split == 'train':
            pkl_path = os.path.join(data_root, 'nuscenes_gen_train.pkl')
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
        elif split == 'val':
            pkl_path = os.path.join(data_root, 'nuscenes_gen_val.pkl')
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
        elif split == 'trainval':
            pkl_path = os.path.join(data_root, 'nuscenes_gen_train.pkl')
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
            pkl_path = os.path.join(data_root, 'nuscenes_gen_val.pkl')
            with open(pkl_path, 'rb') as f:
                data_list += pickle.load(f)
        else:
            raise ValueError(f'invalid data split {split}.')

        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> dict:
        info = self.data_list[index]
        xyzrdm_list = []
        entropy_mask_list = []
        for i in range(self.num_frames):
            xyzrdm = load_points_as_images(
                os.path.join(self.data_root, info['filename'][i]),
                H=self.resolution[0],
                W=self.resolution[1])
            xyzrdm_list.append(xyzrdm)

            entropy = np.fromfile(
                os.path.join(self.uncertainty_root, info['filename'][i]),
                dtype=np.float32)
            indices = np.argsort(entropy)[-self.num_uncertainty:]
            entropy_mask = np.zeros_like(entropy)
            entropy_mask[indices] = 1
            entropy_mask = entropy_mask.reshape(self.resolution[0],
                                                self.resolution[1], 1)
            entropy_mask_list.append(entropy_mask)

        xyzrdm = np.stack(xyzrdm_list, axis=0)
        xyzrdm = xyzrdm.transpose(0, 3, 1, 2)
        xyzrdm *= xyzrdm[:, [5]]

        entropy_mask = np.stack(entropy_mask_list, axis=0)
        entropy_mask = entropy_mask.transpose(0, 3, 1, 2)
        entropy_mask = entropy_mask * xyzrdm[:, [5]]

        return {
            'xyz': xyzrdm[:, :3],
            'reflectance': xyzrdm[:, [3]] / 255.,
            'depth': xyzrdm[:, [4]],
            'mask': xyzrdm[:, [5]],
            'cond_reflectance': xyzrdm[:, [3]] / 255. * entropy_mask,
            'cond_depth': xyzrdm[:, [4]] * entropy_mask,
        }
