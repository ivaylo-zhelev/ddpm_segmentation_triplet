import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import DataLoader, random_split

from torch.optim import Adam, Adagrad, RMSprop, Rprop
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from pandas import DataFrame

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.evaluation import EVAL_FUNCTIONS
from denoising_diffusion_pytorch.loss_functions import (
    mse, triplet_margin_loss, exact_triplet_margin_loss, regularized_triplet_loss, triplet_loss_dynamic_margin)
# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
VALIDATION_FOLDER = "validation"
TESTING_FOLDER = "testing"
GENERATED_FOLDER = "generated"
GT_FOLDER = "ground_truths"
IMAGE_FOLDER = "original_images"
RESULTS_FILE = "evaluation_results.csv"

OPTIMIZERS_DICT = {
    "adam": (Adam, ("lr", "betas")),
    "rprop": (Rprop, ("lr", "etas", "step_sizes")),
    "adagrad": (Adagrad, ("lr", "lr_decay", "weight_decay")),
    "rmsprop": (RMSprop, ("lr", "momentum", "alpha", "weight_decay"))
}

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def split_int_in_propotions(num, split):
    lengths = [math.floor(prop * num) for prop in split]
    remainder = num - sum(lengths)

    ind = 0
    while remainder > 0:
        lengths[ind % len(split)] += 1
        ind += 1
        remainder -= 1

    return lengths

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusionBase(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps = 1000,
        objective = "pred_noise",
        sampling_timesteps = None,
        noising_timesteps = None,
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size
        self.milestone = 0

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        
        assert self.sampling_timesteps <= timesteps
        
        self.noising_timesteps = noising_timesteps or self.sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)


    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        noisy_image_path = self.results_folder / f"noisy_images_{self.milestone}_t={self.sampling_timesteps}_nt={self.noising_timesteps}"
        noisy_image_path.mkdir(exist_ok=True, parents=True)
        
        utils.save_image(x_start[0], noisy_image_path / f"pred_start_{self.milestone}_t={self.sampling_timesteps}_nt={self.noising_timesteps}.png")
        utils.save_image(preds.pred_noise[0], noisy_image_path / f"pred_noise_{self.milestone}_t={self.sampling_timesteps}_nt={self.noising_timesteps}.png")
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, batch_size = 16, imgs = None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), img=imgs)

    @torch.no_grad()
    def p_sample_loop(self, shape, imgs = None, noise = None):
        batch, device = shape[0], self.betas.device

        if img is None:
            img = torch.randn(shape, device=device)
        else:
            t_batched = torch.stack([torch.tensor(self.noising_timesteps, device = device)] * batch)
            noise = default(noise, lambda: torch.randn_like(img))
            imgs = self.q_sample(imgs, t_batched, noise=noise)

        x_start = None

        for t in tqdm(reversed(range(0, self.sampling_timesteps)), desc = 'sampling loop time step', total = self.sampling_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, img = None, clip_denoised = True, noise = None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        t_batched = torch.stack([torch.tensor(self.noising_timesteps, device = device)] * batch)
        if img is None:
            img = torch.randn(shape, device=device)
        else:
            noise = default(noise, lambda: torch.randn_like(img))
            img = self.q_sample(img, t_batched, noise=noise)

        noisy_image_path = self.results_folder / f"noisy_images_{self.milestone}_t={self.sampling_timesteps}_nt={self.noising_timesteps}"
        noisy_image_path.mkdir(exist_ok=True, parents=True)
        utils.save_image(img[0], noisy_image_path / "original.png")
        x_start = None

        for ind, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)
            utils.save_image(pred_noise[0], noisy_image_path / f"pred_noise_{self.milestone}_t={ind}_nt={self.noising_timesteps}.png")
            utils.save_image(x_start[0], noisy_image_path / f"pred_start_{self.milestone}_t={ind}_nt={self.noising_timesteps}.png")
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            utils.save_image(img[0], noisy_image_path / f"denoised_t={ind}.png")
        self.milestone += 1
        img = unnormalize_to_zero_to_one(img)
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class GaussianDiffusion(GaussianDiffusionBase):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            beta_schedule=beta_schedule,
            p2_loss_weight_gamma=p2_loss_weight_gamma,
            p2_loss_weight_k=p2_loss_weight_k,
            ddim_sampling_eta=ddim_sampling_eta
        )
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        
        self.loss_type = loss_type

    
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class GaussianDiffusionSegmentationMapping(GaussianDiffusionBase):
    def __init__(
        self,
        model,
        image_size,
        margin = 1.0,
        regularization_margin = 10.0,
        regularize_to_white_image = True,
        loss_type = "triplet",
        is_loss_time_dependent = False,
        use_ddim_sampling = False,
        *args,
        **kwargs
    ):
        super().__init__(model, image_size, *args, **kwargs)
        self.loss_type = loss_type
        self.margin = margin
        self.regularization_margin = regularization_margin
        self.regularize_to_white_image = regularize_to_white_image
        self.is_loss_time_dependent = is_loss_time_dependent
        self.is_ddim_sampling = self.is_ddim_sampling or use_ddim_sampling

    @property
    def loss_fn(self):
        if self.loss_type == "triplet":
            return triplet_margin_loss
        elif self.loss_type == "mse":
            return mse
        elif self.loss_type == "exact_triplet":
            return exact_triplet_margin_loss
        elif self.loss_type == "triplet_dynamic_margin":
            return triplet_loss_dynamic_margin
        elif self.loss_type == "regularized_triplet":
            return regularized_triplet_loss
        else:
            raise ValueError(f"Loss function of type {self.loss_type} is not supported")

    def p_losses(self, x_start, b_start, t, noise=None):
        if x_start.shape != b_start.shape:
            raise ValueError("The dimensionality of the image and the segmentation maps must be the same")

        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.predict_start_from_noise(x, t, noise=noise)
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        positive, negative = (self.q_sample(x_start=b_start, t=t, noise=noise), x) if self.is_loss_time_dependent \
            else (b_start, x_start)
        loss = self.loss_fn(anchor=model_out,
                            positive=positive,
                            negative=negative,
                            margin=self.margin,
                            regularization_margin=self.regularization_margin,
                            regularize_to_white_image = True,
                            reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    """def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        x_start = self.predict_start_from_noise(x, t, model_output)

        return ModelPrediction(model_output, maybe_clip(x))"""

    def forward(self, sample_pair, *args, **kwargs):
        img, segmentation = torch.unbind(sample_pair, dim=1)
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        _, _, h_segm, w_segm = segmentation.shape
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert h == h_segm and w == w_segm, f"the images and their segmentation must be the same size: {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, segmentation, t, *args, **kwargs)


class Dataset(torch_data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'tif'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class DatasetSegmentation(Dataset):
    def __init__(
        self,
        images_folder,
        segmentations_folder,
        image_mode = "RGB",
        *args,
        **kwargs
    ):
        super().__init__(images_folder, *args, **kwargs)
        self.images_folder = images_folder
        self.segmentations_folder = segmentations_folder
        self.image_mode = image_mode
        segmentation_images = {path.name for path in segmentations_folder.glob("*")}
        self.paths = [
            (path_img, Path(segmentations_folder) / Path(path_img).name)
            for path_img in self.paths if path_img.name in segmentation_images
        ]

    def __getitem__(self, index):
        img_path, segm_path = self.paths[index]
        img = Image.open(img_path).convert(self.image_mode)
        segm = Image.open(segm_path).convert(self.image_mode)
        return torch.stack((self.transform(img), self.transform(segm)), dim=0)


# trainer class
class TrainerBase():
    def __init__(
        self,
        diffusion_model,
        *,
        segmentation_folder = None,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        optimizer = "adam",
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        lr_decay = 0,
        weight_decay = 0,
        rms_prop_alpha = 0.99,
        momentum = 0,
        etas = (0.5, 1.2),
        step_sizes = (1e-06, 50),
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.train_loss_dict = {}
        self.validation_loss_dict = {}

        self.optimizer = optimizer
        self.train_lr = train_lr
        
        self.adam_betas = adam_betas
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.rms_prop_alpha = rms_prop_alpha
        self.momentum = momentum
        self.etas = etas
        self.step_sizes = step_sizes

        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every

        self.augment_horizontal_flip = augment_horizontal_flip
        self.convert_image_to = convert_image_to
        
        self.results_folder = Path(results_folder)
        self.model.results_folder = self.results_folder

    @property
    def IS_SEGMENTATION_TRAINER(self):
        pass

    # dataset and dataloader
    @property
    def ds(self):
        return self._ds
    
    @ds.setter
    def ds(self, ds):
        self._ds = ds
        dl = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        opt_kwargs = {
            "lr": self.train_lr,
            "betas": self.adam_betas,
            "lr_decay": self.lr_decay,
            "etas": self.etas,
            "step_sizes": self.step_sizes,
            "weight_decay": self.weight_decay,
            "alpha": self.rms_prop_alpha,
            "momentum": self.momentum
        }
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in OPTIMIZERS_DICT[self.optimizer][1]}
        try:
            self.opt = OPTIMIZERS_DICT[self.optimizer][0](self.model.parameters(), **opt_kwargs)
        except KeyError:
            assert print(f"{self.optimizer} is not a valid optimizer. Available options are {OPTIMIZERS_DICT.keys()}")

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=self.ema_decay, update_every=self.ema_update_every)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'train_loss_dict': self.train_loss_dict,
            'validation_loss_dict': self.validation_loss_dict,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

        training_loss_df = DataFrame(
            data=[{"epoch": epoch, "loss": loss} for (epoch, loss) in self.train_loss_dict.items()],
            index=list(range(len(self.train_loss_dict))))
        validation_loss_df = DataFrame(
            data=[{"epoch": epoch, "loss": loss} for (epoch, loss) in self.validation_loss_dict.items()],
            index=list(range(len(self.validation_loss_dict))))

        training_loss_df.to_csv(self.results_folder / f'training_loss-{milestone}.csv')
        validation_loss_df.to_csv(self.results_folder / f'validation_loss-{milestone}.csv')

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.train_loss_dict = data['train_loss_dict']
        self.validation_loss_dict = data['validation_loss_dict']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                """grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                print("Total norm:", torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads])))
                for ind, layer in enumerate(self.model.model.ups):
                    grads_layer = [p.grad for p in layer.parameters() if p.grad is not None]
                    print(f"Up layer {ind}:", torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads_layer])))
                for ind, layer in enumerate(self.model.model.downs):
                    grads_layer = [p.grad for p in layer.parameters() if p.grad is not None]
                    print(f"Down layer {ind}:", torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads_layer])))
                    
                grads = [p.grad for p in self.model.model.final_res_block.parameters() if p.grad is not None]
                print("Final resnet:", torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads])))
                grads = [p.grad for p in self.model.model.final_conv.parameters() if p.grad is not None]
                print("Final convolution:", torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads])))
                """
                pbar.set_description(f'Training loss: {total_loss:.4f}')
                self.train_loss_dict[self.step] = total_loss

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()
                    self.validate_or_sample()

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('Training complete!')

    @torch.no_grad()
    def validate_or_sample(self, milestone, device):
        pass

class Trainer(TrainerBase):
    IS_SEGMENTATION_TRAINER = False

    def __init__(
        self,
        diffusion_model,
        folder,
        save_and_sample_every = 1000,
        *args,
        **kwargs
    ):
        super().__init__(diffusion_model, *args, **kwargs)
        self.sample_every = save_and_sample_every
        self.save_every = save_and_sample_every
        self.ds = Dataset(
            folder,
            self.image_size,
            augment_horizontal_flip=self.augment_horizontal_flip,
            convert_image_to=self.convert_image_to
        )

    @torch.no_grad()
    def validate_or_sample(self):
        if self.step != 0 and self.step % self.sample_every == 0:
            milestone = self.step // self.sample_every

            self.ema.ema_model.eval()
            batches = num_to_groups(self.num_samples, self.batch_size)
            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

            for ind, sample in enumerate(all_images_list):
                utils.save_image(
                    sample,
                    self.results_folder / f"sample_{milestone}_{ind}.png")       


class TrainerSegmentation(TrainerBase):
    IS_SEGMENTATION_TRAINER = True

    def __init__(
        self,
        diffusion_model,
        images_folder,
        segmentations_folder,
        validate_every = 1000,
        save_every = 1000,
        data_split = (0.8, 0.1, 0.1),
        eval_metrics = EVAL_FUNCTIONS.keys(),
        seed = 42,
        *args,
        **kwargs
    ):
        super().__init__(diffusion_model, *args, **kwargs)
        self.validate_every = validate_every
        self.save_every = save_every
        self.has_already_validated = False
        self.eval_metrics = eval_metrics

        dataset = DatasetSegmentation(
            images_folder=images_folder,
            segmentations_folder=segmentations_folder,
            image_size=self.image_size,
            augment_horizontal_flip=self.augment_horizontal_flip,
            convert_image_to=self.convert_image_to
        )

        generator = torch.Generator().manual_seed(seed)
        self.ds, self.valid_ds, self.test_ds = random_split(
            dataset,
            lengths=split_int_in_propotions(len(dataset), data_split),
            generator=generator
        )
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count())

        valid_dl = self.accelerator.prepare(valid_dl)
        self.valid_dl = cycle(valid_dl)
        self.test_dl = None

    @torch.no_grad()
    def validate_or_sample(self):
        if self.step != 0 and self.step % self.validate_every == 0:
            validation_set_length = len(self.valid_ds)
            validation_steps = ceil(validation_set_length / self.batch_size)

            self.accelerator.print(f"Validation step {self.step // self.validate_every}...")
            self.ema.ema_model.eval()
            device = self.accelerator.device

            total_loss = 0.0
            for batch_num in tqdm(range(validation_steps), desc="Validation progress:"):
                data = next(self.valid_dl).to(device)

                with self.accelerator.autocast():
                    loss = self.model(data)
                    loss = loss / validation_steps
                    total_loss += loss.item()

                imgs, gt_segm = torch.unbind(data, dim=1)
                noisy_image_path = self.results_folder / f"noisy_images_{self.model.milestone}_t={self.model.sampling_timesteps}_nt={self.model.noising_timesteps}"
                noisy_image_path.mkdir(exist_ok=True, parents=True)
                self.model.milestone += 1
                print(self.model.milestone)
                utils.save_image(gt_segm[0], noisy_image_path / "ground_truth.png")
                self.infer_batch(
                    batch=imgs,
                    results_folder=self.results_folder / VALIDATION_FOLDER / GENERATED_FOLDER / f"epoch_{self.step}",
                    ground_truths_folder=self.results_folder / VALIDATION_FOLDER / GT_FOLDER if not self.has_already_validated else None,
                    original_image_folder=self.results_folder / VALIDATION_FOLDER / IMAGE_FOLDER if not self.has_already_validated else None,
                    ground_truth_segmentation=gt_segm,
                    start_ind=batch_num * self.batch_size,
                    eval_metrics=tuple()
                )

            self.accelerator.print(f'Validation loss: {total_loss:.4f}')
            self.validation_loss_dict[self.step] = total_loss
            self.has_already_validated = True

    @torch.no_grad()
    def test(self, test_ds = None, results_folder = None, eval_metrics = tuple()):
        if test_ds or not self.test_dl:
            test_ds = test_ds or self.test_ds
            test_dl = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=cpu_count())

            test_dl = self.accelerator.prepare(test_dl)
            self.test_dl = cycle(test_dl)

        test_set_length = len(test_ds)
        test_steps = ceil(test_set_length / self.batch_size)

        results_folder = results_folder or self.results_folder

        self.accelerator.print(f"Testing...")
        eval_results = DataFrame()
        device = self.accelerator.device
        for batch_num in tqdm(range(test_steps), desc = "Testing progress:"):
            data = next(self.test_dl).to(device)
            imgs, gt_segm = torch.unbind(data, dim=1)

            eval_results = eval_results.append(
                self.infer_batch(
                    batch=imgs,
                    results_folder=results_folder / TESTING_FOLDER / GENERATED_FOLDER,
                    ground_truths_folder=results_folder / TESTING_FOLDER / GT_FOLDER,
                    original_image_folder=results_folder / TESTING_FOLDER / IMAGE_FOLDER,
                    ground_truth_segmentation=gt_segm,
                    start_ind=batch_num * self.batch_size,
                    eval_metrics=eval_metrics or self.eval_metrics
                )
            )

        eval_results.to_csv(results_folder / TESTING_FOLDER / RESULTS_FILE)
        self.accelerator.print(f"Mean results: \n{eval_results.mean(numeric_only=True)}")

    @torch.no_grad()
    def infer_batch(
        self,
        batch,
        ground_truth_segmentation = None,
        results_folder = None,
        ground_truths_folder = None,
        original_image_folder = None,
        threshold = 0.5,
        start_ind = 0,
        eval_metrics = EVAL_FUNCTIONS.keys()
    ):
        results_folder = results_folder or self.results_folder
        results_folder.mkdir(exist_ok=True, parents=True)
        if ground_truths_folder:
            ground_truths_folder.mkdir(exist_ok=True, parents=True)
        if original_image_folder:
            original_image_folder.mkdir(exist_ok=True, parents=True)

        pred_segmentations = self.ema.ema_model.sample(batch_size=batch.shape[0], imgs=batch)
        imgs_list = list(torch.unbind(batch))
        segm_list = list(torch.unbind(pred_segmentations))
        gt_list = list(torch.unbind(ground_truth_segmentation)) if ground_truth_segmentation is not None \
            else [None] * len(imgs_list)

        eval_results = DataFrame()
        for ind, (image, segmentation, ground_truth) in enumerate(zip(imgs_list, segm_list, gt_list)):
            segmentation_filename = results_folder / f"sample_{start_ind + ind}.png"
            ground_truth_filename = None
            original_image_filename = None

            if ground_truths_folder and ground_truth is not None:
                ground_truth_filename = ground_truths_folder / f"sample_{start_ind + ind}.png"
                utils.save_image(
                    ground_truth,
                    ground_truth_filename)

            if original_image_folder:
                original_image_filename = original_image_folder / f"sample_{start_ind + ind}.png"
                utils.save_image(
                    image,
                    original_image_filename)

            utils.save_image(
                segmentation,
                segmentation_filename)  

            if eval_metrics and ground_truth_segmentation is not None:
                image_info = {
                    "predicted_segmentation": segmentation_filename,
                    "ground_truth_segmentation": ground_truth_filename,
                    "original_image": original_image_filename
                }
                eval_results = eval_results.append(
                    self.evaluate(
                        predicted=segmentation,
                        ground_truth=ground_truth,
                        image_info=image_info,
                        metrics=eval_metrics,
                        index=ind,
                        threshold=threshold
                    )
                )

        return eval_results

    @torch.no_grad()
    def infer_folder(
        self,
        folder_path,
        results_path,
        image_size = None,
        batch_size = None,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'tif']
    ):
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)

        batch_size = batch_size or self.batch_size

        dataset = Dataset(folder_path, image_size or self.image_size, exts)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cpu_count())

        data_loader = self.accelerator.prepare(data_loader)
        data_loader = cycle(data_loader)

        total_batches = ceil(len(dataset) / batch_size)
        for batch_num in tqdm(range(total_batches), desc=f"Total number of batches: {total_batches}"):
            batch = next(data_loader).to(self.accelerator.device)
            
            self.infer_batch(
                batch,
                results_folder=results_path / GENERATED_FOLDER,
                original_image_folder=results_path / IMAGE_FOLDER,
                eval_metrics = tuple()
            )

    @torch.no_grad()
    def infer_image(self, image_path, results_path):
        image = Image.open(image_path)
        transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])
        image = torch.unsqueeze(transform(image), dim=0).to(self.accelerator.device)

        segmentation = self.ema.ema_model.sample(batch_size=1, imgs=image)
        utils.save_image(segmentation, results_path)

    @staticmethod
    @torch.no_grad()
    def evaluate(
        predicted,
        ground_truth,
        image_info,
        metrics = EVAL_FUNCTIONS.keys(),
        index = 0,
        threshold = 0.5
    ):
        predicted = torch.unsqueeze(predicted, dim=0)
        ground_truth = torch.unsqueeze(ground_truth, dim=0)

        eval_dict = {key: value for (key, value) in image_info.items() if value}
        for metric in metrics:
            try:
                eval_dict[metric] = EVAL_FUNCTIONS[metric](predicted, ground_truth, threshold=threshold)
            except KeyError:
                raise ValueError(
                    f"Metric {metric} is not a valid metric. Options are: {EVAL_FUNCTIONS.keys()}")

        return DataFrame(eval_dict, index=[index])