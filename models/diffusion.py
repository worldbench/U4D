import math
from functools import partial
from typing import List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm


def _log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))


def _log_snr_schedule_linear(t: torch.Tensor) -> torch.Tensor:
    return -_log(expm1(1e-4 + 10 * (t**2)))[:, None, None, None, None]


def _log_snr_schedule_cosine(t: torch.Tensor,
                             logsnr_min: float = -15,
                             logsnr_max: float = 15) -> torch.Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * _log(torch.tan(t_min + t *
                               (t_max - t_min)))[:, None, None, None, None]


def _log_snr_schedule_cosine_shifted(t: torch.Tensor,
                                     image_d: float,
                                     noise_d: float,
                                     logsnr_min: float = -15,
                                     logsnr_max: float = 15) -> torch.Tensor:
    log_snr = _log_snr_schedule_cosine(
        t, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    shift = 2 * math.log(noise_d / image_d)
    return log_snr + shift


def _log_snr_schedule_cosine_interpolated(
        t: torch.Tensor,
        image_d: float,
        noise_d_low: float,
        noise_d_high: float,
        logsnr_min: float = -15,
        logsnr_max: float = 15) -> torch.Tensor:
    logsnr_low = _log_snr_schedule_cosine_shifted(t, image_d, noise_d_low,
                                                  logsnr_min, logsnr_max)
    logsnr_high = _log_snr_schedule_cosine_shifted(t, image_d, noise_d_high,
                                                   logsnr_min, logsnr_max)
    return t * logsnr_low + (1 - t) * logsnr_high


def _log_snr_to_alpha_sigma(
        log_snr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
    return alpha, sigma


class GaussianDiffusion(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 prediction_type: str = 'eps',
                 loss_type: Union[str, nn.Module] = 'l2',
                 noise_schedule: str = 'cosine',
                 min_snr_loss_weight: bool = True,
                 min_snr_gamma: float = 5.0,
                 clip_sample: bool = True,
                 clip_sample_range: float = 1,
                 image_d: Optional[float] = None,
                 noise_d_low: float = None,
                 noise_d_high: float = None) -> None:
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.objective = prediction_type
        self.noise_schedule = noise_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.criterion == nn.SmoothL1Loss(reduction='none')
        elif isinstance(loss_type, nn.Module):
            self.criterion = loss_type
        else:
            raise ValueError(f'invalid criterion: {loss_type}')

        if hasattr(self.criterion, 'reduction'):
            assert self.criterion.reduction == 'none'

        assert hasattr(self.model, 'resolution')
        assert hasattr(self.model, 'in_channels')
        assert hasattr(self.model, 'num_frames')
        self.sampling_shape = (self.model.num_frames, self.model.in_channels,
                               *self.model.resolution)

        self.image_d = image_d
        self.noise_d_low = noise_d_low
        self.noise_d_high = noise_d_high

        self.setup_parameters()
        self.register_buffer('_dummy', torch.tensor([]))

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def randn(self,
              *shape,
              rng: Optional[Union[List[torch.Generator],
                                  torch.Generator]] = None,
              **kwargs) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng])
        else:
            raise ValueError(f'invalid rng: {rng}')

    def randn_like(
        self,
        x: torch.Tensor,
        rng: Optional[Union[List[torch.Generator], torch.Generator]] = None
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        if self.noise_schedule == 'linear':
            self.log_snr = _log_snr_schedule_linear
        elif self.noise_schedule == 'cosine':
            self.log_snr = _log_snr_schedule_cosine
        elif self.noise_schedule == 'cosine_shifted':
            assert self.image_d is not None and self.noise_d_low is not None
            self.log_snr = partial(
                _log_snr_schedule_cosine_shifted,
                image_d=self.image_d,
                noise_d=self.noise_d_low)
        elif self.noise_schedule == 'cosine_interpolated':
            assert (self.image_d is not None and self.noise_d_low is not None
                    and self.noise_d_high is not None)
            self.log_snr = partial(
                _log_snr_schedule_cosine_interpolated,
                image_d=self.image_d,
                noise_d_low=self.noise_d_low,
                noise_d_high=self.noise_d_high)
        else:
            raise ValueError(f'invalid beta schedule: {self.noise_schedule}')

    def sample_timesteps(self, batch_size: int,
                         device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_network_condition(self, steps: torch.Tensor) -> torch.Tensor:
        return self.log_snr(steps)[:, 0, 0, 0, 0]

    def get_target(self, x_0: torch.Tensor, step_t: torch.Tensor,
                   noise: torch.Tensor) -> torch.Tensor:
        if self.objective == 'eps':
            target = noise
        elif self.objective == 'x_0':
            target = x_0
        elif self.objective == 'v':
            log_snr = self.log_snr(step_t)
            alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
            target = alpha * noise - sigma * x_0
        else:
            raise ValueError(f'invalid objective: {self.objective}')
        return target

    def get_loss_weight(self, steps: torch.Tensor) -> torch.Tensor:
        log_snr = self.log_snr(steps)
        snr = log_snr.exp()
        clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            clipped_snr.clamp_(max=self.min_snr_gamma)
        if self.objective == 'eps':
            loss_weight = clipped_snr / snr
        elif self.objective == 'x_0':
            loss_weight = clipped_snr
        elif self.objective == 'v':
            loss_weight = clipped_snr / (snr + 1)
        else:
            raise ValueError(f'invalid objective: {self.objective}')
        return loss_weight

    @autocast(device_type='cuda', enabled=False)
    def q_step_from_x_0(
        self,
        x_0: torch.Tensor,
        step_t: torch.Tensor,
        rng: Optional[Union[List[torch.Generator], torch.Generator]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward diffusion process q(zt|x0) where 0<t<1
        noise = self.randn_like(x_0, rng=rng)
        log_snr = self.log_snr(step_t)
        alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
        x_t = x_0 * alpha + noise * sigma
        return x_t, noise

    def p_loss(self,
               x_0: torch.Tensor,
               steps: torch.Tensor,
               cond: Optional[torch.Tensor] = None,
               loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
        x_t, noise = self.q_step_from_x_0(x_0, steps)
        timesteps = self.get_network_condition(steps)
        prediction = self.model(x_t, timesteps, cond)
        target = self.get_target(x_t, steps, noise)
        loss = self.criterion(prediction, target)  # (B,T,C,H,W)
        loss = einops.reduce(loss * loss_mask, 'B ... -> B ()', 'sum')
        loss_mask = einops.reduce(loss_mask, 'B ... -> B ()', 'sum')
        loss = loss / loss_mask.add(1e-8)  # (B,)
        loss = (loss * self.get_loss_weight(steps)).mean()
        return loss

    def forward(self,
                x_0: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        loss = self.p_loss(x_0, steps, cond, loss_mask)
        return loss

    @torch.inference_mode()
    def p_step(
        self,
        x_t: torch.Tensor,
        step_t: torch.Tensor,
        step_s: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        rng: Optional[Union[List[torch.Generator], torch.Generator]] = None
    ) -> torch.Tensor:
        # reverse diffusion process q(zs|zt) where 0<s<t<1
        log_snr_t = self.log_snr(step_t)
        log_snr_s = self.log_snr(step_s)
        alpha_t, sigma_t = _log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = _log_snr_to_alpha_sigma(log_snr_s)
        prediction = self.model(x_t, log_snr_t[:, 0, 0, 0, 0], cond)
        if self.objective == 'eps':
            x_0 = (x_t - sigma_t * prediction) / alpha_t
        elif self.objective == 'v':
            x_0 = alpha_t * x_t - sigma_t * prediction
        elif self.objective == 'x_0':
            x_0 = prediction
        else:
            raise ValueError(f'invalid objective: {self.objective}')

        if self.clip_sample:
            x_0.clamp_(-self.clip_sample_range, self.clip_sample_range)

        c = -expm1(log_snr_t - log_snr_s)
        mean = alpha_s * (x_t * (1 - c) / alpha_t + c * x_0)
        std = sigma_s * c.sqrt()
        noise = self.randn_like(x_t, rng=rng)
        x_s = mean + std * noise
        return x_s

    @torch.inference_mode()
    def sample(self,
               batch_size: int,
               num_steps: int,
               cond: Optional[torch.Tensor] = None,
               progress: bool = True,
               rng: Optional[Union[List[torch.Generator],
                                   torch.Generator]] = None,
               return_all: bool = False) -> torch.Tensor:
        x = self.randn(
            batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        p_step_kwargs = dict(rng=rng, cond=cond)
        tqdm_kwargs = dict(desc='sampling', leave=False, disable=not progress)
        for i in tqdm(range(num_steps), **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            x = self.p_step(x, step_t, step_s, **p_step_kwargs)
            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x
