from enum import Enum
from typing import Literal

import torch

from siscos_nodes.src.siscos_nodes.segmentation.clipseg import CLIPSegSegmentationModel
from siscos_nodes.src.siscos_nodes.segmentation.groupvit import (
    GroupVitSegmentationModel,
)
from siscos_nodes.src.siscos_nodes.segmentation.segmentation_model import (
    SegmentationModel,
)
from siscos_nodes.src.siscos_nodes.util.tensor_common import (
    threshold_and_normalize_tensor,
)


class ESegmentationModel(str, Enum):
    """Defines the available image-segmentation models."""
    CLIPSEG = "clipseg"
    GROUP_VIT = "groupvit"
    GROUNDED_SAM = "grounded_sam2"  # TODO: Implement this model

class EMixingMode(str, Enum):
    """Defines the blending modes for segmentation masks."""
    SUPPRESS = "suppress"
    SUBTRACT = "subtract"
    ADD = "add"
    MULTIPLY = "multiply"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    XOR = "xor"
    OR = "or"
    AND = "and"

# TODO:(sisco): Figure out a way to use the enums in the type hints for the input fields instead of these gross literals...
MixingMode = Literal[EMixingMode.SUPPRESS, EMixingMode.SUBTRACT, EMixingMode.ADD, EMixingMode.MULTIPLY, EMixingMode.AVERAGE, EMixingMode.MAX, EMixingMode.MIN, EMixingMode.XOR, EMixingMode.OR, EMixingMode.AND]
SegmentationModelType = Literal[ESegmentationModel.CLIPSEG, ESegmentationModel.GROUP_VIT]
SEGMENTATION_MODEL_TYPES: dict[ESegmentationModel, type[SegmentationModel]] = {
    ESegmentationModel.CLIPSEG: CLIPSegSegmentationModel,
    ESegmentationModel.GROUP_VIT: GroupVitSegmentationModel,
}

def compare_scalar_fields(mode: EMixingMode, lhs: torch.Tensor, rhs: torch.Tensor, rhs_factor: float) -> torch.Tensor:
    """Blend two tensors holding 2D scalar fields together using the specified mode.
        For non-binary modes, the factor is used to scale the rhs tensor before blending.
        For binary modes (OR/AND/XOR) the factor acts as an offset for rhs, allowing it to still be used to adjust the outcome.
    """
    assert lhs.dim() == rhs.dim() == 4, f"Expected tensors to have shape [B, C, H, W], but got {lhs.shape} and {rhs.shape}"
    match mode:
        case EMixingMode.SUPPRESS:
            # suppress the positive mask according to the inverse of the negative mask
            inv = torch.sub(1.0, rhs, alpha=rhs_factor)
            return torch.mul(lhs, inv)
        case EMixingMode.SUBTRACT:
            return torch.sub(lhs, rhs, alpha=rhs_factor)
        case EMixingMode.ADD:
            return torch.add(lhs, rhs, alpha=rhs_factor)
        case EMixingMode.MULTIPLY:
            return torch.mul(lhs, (rhs * rhs_factor))
        case EMixingMode.AVERAGE:
            # average the two masks together
            return torch.mean(torch.cat((lhs, rhs * rhs_factor), dim=1), dim=1, keepdim=True)
        case EMixingMode.MAX:
            # take the maximum of the two masks
            return torch.max(lhs, (rhs * rhs_factor))
        case EMixingMode.MIN:
            # take the minimum of the two masks
            return torch.min(lhs, (rhs * rhs_factor))
        case EMixingMode.XOR:
            orig_type = lhs.dtype
            return torch.logical_xor(lhs.gt(0.0), torch.gt(rhs, 1.0 - rhs_factor)).to(orig_type)
        case EMixingMode.OR:
            orig_type = lhs.dtype
            return torch.logical_or(lhs.gt(0.0), torch.gt(rhs, 1.0 - rhs_factor)).to(orig_type)
        case EMixingMode.AND:
            orig_type = lhs.dtype
            return torch.logical_and(lhs.gt(0.0), torch.gt(rhs, 1.0 - rhs_factor)).to(orig_type)
        case _:
            raise ValueError(f"Unknown blend mode: {mode}")

def collapse_scalar_fields(tensor: torch.Tensor, threshold: float, blend_mode: EMixingMode) -> torch.Tensor:
    """Collapse all 2D scalar fields in the tensor into a single layer using the specified blending mode."""
    # For these blending modes, we apply the logic across the channel dimension.
    # This is done to combine the results of multiple prompts into a single mask.
    assert tensor.dim() == 4, f"Expected tensor to have 4 dimensions, but got {tensor.ndim}"
    # assert tensor.dim() == 4, f"Expected tensor to have shape [B, C, H, W], but got {tensor.shape}"
    match blend_mode:
        case EMixingMode.AVERAGE:
            tensor = tensor.mean(dim=1, keepdim=True).unsqueeze(0)
        case EMixingMode.SUPPRESS:
            if (tensor.shape[1] > 1):
                tensor = tensor.clamp(min=0.0)
                left = tensor[0:, 0:1]
                right = tensor[0:, 1:].amax(dim=1, keepdim=True)
                tensor = left * (1 - right)
        case EMixingMode.SUBTRACT:
            # Subtract all subsequent layers from the first
            if (tensor.shape[1] > 1):
                left = tensor[0:, 0:1]
                right = tensor[0:, 1:].sum(dim=1, keepdim=True)
                tensor =  left.subtract(right).clamp(min=0.0) # results in [N-1, H, W]
        case EMixingMode.ADD:
            # Add all the batches together
            tensor = tensor.sum(dim=1, keepdim=True)
        case EMixingMode.MULTIPLY:
            # Multiply all the layers together
            tensor = tensor.prod(dim=1, keepdim=True)
        case EMixingMode.MAX:
            # Take the maximum of all the batches
            tensor = tensor.amax(dim=1, keepdim=True)
        case EMixingMode.MIN:
            # Take the minimum of all the batches
            tensor = tensor.amin(dim=1, keepdim=True)
        case EMixingMode.XOR:
            # Take the exclusive or of all the batches
            orig_type = tensor.dtype
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_xor(tensor.amax(dim=1)).to(orig_type)
        case EMixingMode.OR:
            # Take the logical or of all the batches
            orig_type = tensor.dtype
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_or(tensor.amax(dim=1)).to(orig_type)
        case EMixingMode.AND:
            # Take the logical and of all the batches
            orig_type = tensor.dtype
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_and(tensor.amax(dim=1)).to(orig_type)
        case _:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

    assert tensor.shape[0] == 1 and tensor.ndim == 4, f"Expected tensor to have shape [1, 1, H, W], but got {tensor.shape}"
    # Normalize the tensor to be between 0 and 1
    result = threshold_and_normalize_tensor(tensor, threshold)

    assert result.shape[0] == 1 and result.ndim == 4, f"Expected tensor to have shape [1, 1, H, W], but got {result.shape}"
    return result
