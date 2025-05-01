import math
from functools import lru_cache

import torch
import torchvision
import torchvision.transforms.functional


# @torch.compile(dynamic=True)
@torch.no_grad()
@torch.jit.script
def resize_tensor(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
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
@torch.no_grad()
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
@torch.no_grad()
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
@torch.no_grad()
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

@torch.no_grad()
@torch.jit.script
def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize each H/W layer(separately) of the tensor to the range [0, 1]."""
    ih = tensor.dim() - 2
    iw = ih + 1
    vmin = tensor.amin(dim=(ih, iw), keepdim=True)
    vmax = tensor.amax(dim=(ih, iw), keepdim=True)

    # if a layer is zero-filled, we want to ensure the normalized result is also zero.
    # logically, the layer is zeroed only if the min and max are both equal to zero
    # boolean logic deduction shows that this is equivalent to: !(vmin != 0 | vmax != 0)
    # so we use (vmin != 0 | vmax != 0) to get a boolean tensor of the same shape as vmin/vmax, and then later multiply the tensor by it to zero out the null layers.
    layer_is_non_zero = torch.logical_or(vmin, vmax)

    # avoid division by zero for layers filled with a single constant value by adding a small epsilon to the range
    eps = 1e-06 # torch.finfo(tensor.dtype).eps # e.g. ~1.19e-07 for float32
    range = (vmax - vmin) + eps

    # add epsilon to cause layers with a constant value to result in 1.0 after normalization
    result = (tensor - vmin) + eps 
    # zero out the null layers
    result *= layer_is_non_zero
    # normalize each [H,W] slice to [0,1]
    return result / range

@torch.no_grad()
@torch.jit.script
def threshold_and_normalize_tensor(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Normalize each H/W layer(separately) of the tensor to the range [0, 1], then offset by the threshold and renormalize."""
    ih = tensor.dim() - 2
    iw = ih + 1
    vmin = tensor.amin(dim=(ih, iw), keepdim=True)
    vmax = tensor.amax(dim=(ih, iw), keepdim=True)

    # if a layer is zero-filled, we want to ensure the normalized result is also zero.
    # logically, the layer is zeroed only if the min and max are both equal to zero
    # boolean logic deduction shows that this is equivalent to: !(vmin != 0 | vmax != 0)
    # so we use (vmin != 0 | vmax != 0) to get a boolean tensor of the same shape as vmin/vmax, and then later multiply the tensor by it to zero out the null layers.
    layer_is_non_zero = torch.logical_or(vmin, vmax)

    # avoid division by zero for layers filled with a single constant value by adding a small epsilon to the range
    eps = 1e-06 # torch.finfo(tensor.dtype).eps # e.g. ~1.19e-07 for float32
    range = (vmax - vmin) + eps

    # add epsilon to cause layers with a constant value to result in 1.0 after normalization
    result = (tensor - vmin) + eps 
    # zero out the null layers
    result *= layer_is_non_zero
    # normalize each [H,W] slice to [0,1]
    result /= range
    inv = 1.0 / ((1.0 - threshold) + eps)
    return (result - threshold).mul(inv).clamp(min=0.0, max=1.0)

@torch.no_grad()
@torch.jit.script
def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize each H/W layer(separately) of the logits to the range [-1, 1]"""
    assert logits.ndim == 3, "Input tensor must be 3D (batch, height, width)."
    # compute per‐slice min and max, keeping dims so they broadcast over H,W
    ih = logits.dim() - 2
    iw = ih + 1
    vmin = logits.amin(dim=(ih, iw), keepdim=True)   # shape [N,1,1]
    vmax = logits.amax(dim=(ih, iw), keepdim=True)   # shape [N,1,1]

    # avoid division by zero for layers filled with a single constant value by adding a small epsilon to the range
    eps = 1e-06 # torch.finfo(tensor.dtype).eps # e.g. ~1.19e-07 for float32
    range = (vmax - vmin) + eps

    # normalize each [H,W] slice to [0,1]
    result = (logits - vmin) + eps
    result /= range
    return (result * 2.0) - 1.0 # scale to [-1, 1]
