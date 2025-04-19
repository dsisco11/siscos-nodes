import math

import torch


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

    return torch.nn.functional.interpolate(tensor, size=target_size, mode='bicubic', align_corners=True)

@torch.jit.script
def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a 4D torch tensor (batch, channels, height, width).

    Args:
        tensor (torch.Tensor): Input tensor to blur.
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Blurred tensor.
    """
    # Calculate ideal kernel size based on sigma
    # radius = ceil(3 * sigma)
    # kernel_size = 2 * radius + 1 (to make it odd)
    radius = math.ceil(3 * sigma)
    kernel_size = 2 * radius + 1

    # create x on the right device + dtype in one go
    x = torch.arange(kernel_size,
                     device=tensor.device,
                     dtype=tensor.dtype) - (kernel_size // 2)
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # outer product → 2D kernel, still (k,k)
    kernel2d = gauss[:, None] * gauss[None, :]

    # now repeat into a real contiguous weight tensor of shape (C,1,k,k)
    C = tensor.shape[1]
    weight = kernel2d.unsqueeze(0).unsqueeze(0)           # (1,1,k,k)
    weight = weight.repeat(C, 1, 1, 1).contiguous()       # (C,1,k,k)

    # depthwise conv
    padding = kernel_size // 2

    # Apply Gaussian blur using conv2d
    padding = kernel_size // 2
    return torch.nn.functional.conv2d(tensor, weight, padding=padding, groups=C)

@torch.jit.script
def apply_feathering_square(tensor: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Apply a (square) dilation to the mask tensor by 'radius' pixels,
    i.e. expand any non-zero region outward.
    Supports 2D ([H,W]), 3D ([C,H,W]) and 4D ([N,C,H,W]) inputs.
    """
    # convert float radius → integer radius
    # (anything <0.5 → radius=0 → no-op)
    int_radius: int = int(radius)
    if int_radius < 1:
        return tensor

    # kernel size = 2*radius + 1 (odd)
    k: int = int_radius * 2 + 1

    # reshape to 4D if needed
    if tensor.dim() == 2:
        # [H, W] → [1,1,H,W]
        x = tensor.unsqueeze(0).unsqueeze(0)
        y = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=int_radius)
        return y.squeeze(0).squeeze(0)

    elif tensor.dim() == 3:
        # [C, H, W] → [1,C,H,W]
        x = tensor.unsqueeze(0)
        y = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=int_radius)
        return y.squeeze(0)

    elif tensor.dim() == 4:
        # [N,C,H,W] — apply directly
        return torch.nn.functional.max_pool2d(tensor, kernel_size=k, stride=1, padding=int_radius)

    else:
        # unsupported rank: just return unchanged
        return tensor

@torch.jit.script
def apply_feathering_ellipse(tensor: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Soft circular/elliptical dilation of a mask:
    - Any positive value in `tensor` will bleed outward up to `radius` pixels
      with a linear fall‑off (1 at center → 0 at boundary).
    - Works on [H,W], [C,H,W] or [N,C,H,W].
    """
    # integer radius
    ir: int = int(radius)
    if ir < 1:
        return tensor

    # kernel size
    k: int = ir * 2 + 1

    # build 2D fall‑off kernel
    # shape [k,k], dtype/device match input
    cfg = tensor.new_zeros([k, k])
    for i in range(k):
        for j in range(k):
            dx: float = float(i - ir)
            dy: float = float(j - ir)
            # actual Euclidean distance
            d: float = (dx * dx + dy * dy) ** 0.5
            if d <= radius:
                cfg[i, j] = 1.0 - (d / radius)
            else:
                cfg[i, j] = 0.0

    # normalize kernel so full‑mask stays at 1.0 after conv
    total = cfg.sum()
    if total > 0.0:
        cfg = cfg / total

    # shape‑helpers: promote to 4D
    x: torch.Tensor = tensor
    if tensor.dim() == 2:
        x = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        x = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        x = tensor
    else:
        # unsupported rank
        return tensor

    n, c, h, w = x.shape

    # make depth‑wise conv weight [C,1,k,k]
    # each channel uses the same cfg
    wght = cfg.view(1, 1, k, k).repeat(c, 1, 1, 1)

    # convolve with padding so output is same H×W
    y = torch.nn.functional.conv2d(x, wght, bias=None, stride=1, padding=ir, groups=c)

    # squeeze back to original rank
    if tensor.dim() == 2:
        return y.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        return y.squeeze(0)
    else:
        return y

@torch.jit.script
def box_blur(tensor: torch.Tensor, radius: float) -> torch.Tensor:
    kernel = torch.ones((1, 1, int(radius), int(radius)), device=tensor.device, dtype=tensor.dtype)
    kernel = kernel / (radius * radius)
    return torch.nn.functional.conv2d(tensor, kernel, padding=(int(radius) // 2, int(radius) // 2), groups=1)

@torch.jit.script
def scale_logits(logits: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale the logits by a given factor.
    Args:
        logits (torch.Tensor): The input logits tensor.
        scale (float): The scaling factor.
    """
    return (logits / scale).softmax(dim=2)

@torch.jit.script
def normalize_tensor(tensor: torch.Tensor, min_threshold: float = 0.0) -> torch.Tensor:
    """Normalize the tensor to the range [0, 1]."""
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

def print_tensor_stats(tensor: torch.Tensor, name: str = "Tensor"):
    """Print the statistics of a tensor."""
    print(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}")