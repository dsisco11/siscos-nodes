import math
from functools import lru_cache

import torch
import torchvision
import torchvision.transforms.functional


# @torch.compile(dynamic=True)
@torch.jit.script
def upscale_tensor(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Upscale a 4D torch tensor (batch, channels, height, width) to a given size.

    Args:
        tensor (torch.Tensor): Input tensor to upscale.
        target_size (tuple[int, int]): Target size as (height, width).

    Returns:
        torch.Tensor: Upscaled tensor.
    """
    if tensor.dim() != 4:
        raise ValueError("Input tensor must be 4D (batch, channels, height, width).")

    return torch.nn.functional.interpolate(tensor, size=target_size, mode='bilinear', align_corners=True)

# @torch.compile(dynamic=True)
@torch.jit.script
def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a 4D torch tensor (batch, channels, height, width).

    Args:
        tensor (torch.Tensor): Input tensor to blur.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Blurred tensor.
    """
    # kernel_size = 2*ceil(3σ)+1
    radius = math.ceil(3 * sigma)
    kernel_size = 2 * radius + 1
    return torchvision.transforms.functional.gaussian_blur(tensor, [kernel_size, kernel_size], [sigma, sigma])

@lru_cache(maxsize=8)
def make_ellipse_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a flattened 2D linear fall‑off kernel of size (k*k)."""
    k = radius*2 + 1
    # vectorized radius map
    coords = torch.arange(k, device=device, dtype=dtype) - radius
    xx = coords[:, None].pow(2)
    yy = coords[None, :].pow(2)
    d = (xx + yy).sqrt()
    # linear fall‑off
    w = (1.0 - d / float(radius)).clamp(min=0)
    return w.view(-1)  # shape [k*k]

# @torch.compile(dynamic=True)
@torch.jit.script
def apply_feathering_ellipse(
    tensor: torch.Tensor,
    radius: float
) -> torch.Tensor:
    """
    Soft circular/elliptical dilation of a mask via weighted max‑filter.
    """
    ir = int(radius)
    if ir < 1:
        return tensor

    # get flattened linear kernel [k*k], no normalization
    device, dtype = tensor.device, tensor.dtype
    flatw = make_ellipse_kernel(ir, device, dtype)  # [k*k]
    k = ir*2 + 1

    # promote to NCHW
    x = tensor
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    # now x is [N, C, H, W]

    N, C, H, W = x.shape

    # extract all k×k patches → [N, C*k*k, H*W]
    patches = torch.nn.functional.unfold(x, kernel_size=k, padding=ir)
    # reshape → [N, C, k*k, H*W]
    patches = patches.view(N, C, k*k, H*W)

    # weight each element in the window
    # flatw: [k*k] → [1,1,k*k,1]
    patches = patches * flatw.view(1,1,-1,1)

    # take the max over the window → [N, C, H*W]
    out = patches.max(dim=2).values

    # reshape back → [N, C, H, W]
    out = out.view(N, C, H, W)

    # restore original rank
    if tensor.ndim == 2:
        return out[0,0]
    if tensor.ndim == 3:
        return out[0]
    return out

# @torch.jit.script
def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize each H/W layer(separately) of the tensor to the range [0, 1]."""
    ih = tensor.dim() - 2
    iw = ih + 1
    vmin = tensor.amin(dim=(ih, iw), keepdim=True)
    vmax = tensor.amax(dim=(ih, iw), keepdim=True)
    # avoid division by zero if a slice is constant
    range = vmax - vmin
    range[range == 0] = 1e-6
    # return ((tensor - vmin) / range)
    return torch.where(
        range != 0,
        (tensor - vmin) / range,
        torch.ones_like(tensor)
    )

@torch.jit.script
def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize each H/W layer(separately) of the logits to the range [-1, 1]"""
    assert logits.ndim == 3, "Input tensor must be 3D (batch, height, width)."
    # compute per‐slice min and max, keeping dims so they broadcast over H,W
    vmin = logits.amin(dim=(1,2), keepdim=True)   # shape [N,1,1]
    vmax = logits.amax(dim=(1,2), keepdim=True)   # shape [N,1,1]

    # avoid division by zero if a slice is constant
    range = vmax - vmin
    range[range == 0] = 1e-6

    # normalize each [H,W] slice to [0,1]
    norm = (logits - vmin) / range
    return (norm * 2.0) - 1.0 # scale to [-1, 1]

def print_tensor_stats(tensor: torch.Tensor, name: str = "Tensor"):
    """Print the statistics of a tensor."""
    print(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}")
