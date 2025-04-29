from enum import Enum
from typing import Any, Literal

import torch
from invokeai.app.invocations.baseinvocation import (
    BaseInvocationOutput,
    InvocationContext,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, OutputField
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_tensor as image_to_tensor


class EMaskingMode(str, Enum):
    """Enum for Masking Modes, which determine the form of a given image mask."""

    BOOLEAN = "Boolean" # eg: boolean tensor, shape: [1, H, W], dtype: bool
    GRADIENT = "Gradient" # eg: scalar(0.0 - 1.0) tensor, shape: [1, H, W], dtype: float8/16/32
    IMAGE_ALPHA = "image_alpha" # eg: image mode='RGBA', dtype: uint8, alpha channel is the mask
    IMAGE_COMPOUND = "image_compound"# eg: each channel is a different mask
    IMAGE_LUMINANCE = "image_luminance" # eg: image mode='L', dtype: uint8

LMaskingMode = Literal[EMaskingMode.BOOLEAN, EMaskingMode.GRADIENT, EMaskingMode.IMAGE_ALPHA, EMaskingMode.IMAGE_COMPOUND, EMaskingMode.IMAGE_LUMINANCE]

class MaskingField(BaseModel):
    """A masking primitive field."""

    asset_id: str = Field(description="The id/name of the mask image within the asset cache system.")
    mode: EMaskingMode = Field(description="The masking mode specifies how the mask is represented.")

    def __init__(self, mode: EMaskingMode, asset_id: str | None = None, image_name: str | None = None, tensor_name: str | None = None):
        """Initialize the MaskingField with an asset ID and mode."""
        if (asset_id is None and image_name is None and tensor_name is None):
            raise ValueError("Either asset_id, image_name or tensor_name must be provided.")
        
        # Use the first non-None value
        _id: str = asset_id or image_name or tensor_name # type: ignore
        if not _id:
            raise ValueError("Either asset_id, image_name or tensor_name must be provided.")
        
        super().__init__(asset_id=_id, mode=mode)

    def load(self, context: InvocationContext) -> torch.Tensor:
        """Load the mask from the asset cache."""
        match (self.mode):
            case EMaskingMode.BOOLEAN | EMaskingMode.GRADIENT:
                return context.tensors.load(self.asset_id)
            case EMaskingMode.IMAGE_LUMINANCE:
                return image_to_tensor(context.images.get_pil(self.asset_id, mode='L'))
            case EMaskingMode.IMAGE_ALPHA:
                return image_to_tensor(context.images.get_pil(self.asset_id, mode='RGBA')).split(1)[-1]
            case EMaskingMode.IMAGE_COMPOUND:
                return image_to_tensor(context.images.get_pil(self.asset_id, mode='RGBA'))
            case _:
                raise ValueError(f"Unsupported mask mode: {self.mode}")


@invocation_output("masking_node_output")
class MaskingNodeOutput(BaseInvocationOutput):
    mask: MaskingField = OutputField(title="Mask")

@invocation_output("adv_mask_output")
class AdvancedMaskOutput(BaseInvocationOutput):

    mask: MaskingField = OutputField(description="The mask.")
    image: ImageField = OutputField(description="The mask image.")
    width: int = OutputField(description="The width of the mask in pixels.")
    height: int = OutputField(description="The height of the mask in pixels.")
