from enum import Enum

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import BaseInvocationOutput, invocation_output
from invokeai.app.invocations.fields import ImageField, OutputField


class EMaskMode(str, Enum):
    """Enum for Masking Modes, which determine the form of a given image mask."""

    BOOLEAN = "boolean" # eg: boolean tensor, shape: [1, H, W], dtype: bool
    SCALAR = "scalar" # eg: scalar(0.0 - 1.0) tensor, shape: [1, H, W], dtype: float8/16/32
    UINT8 = "uint8" # eg: uint8 tensor, shape: [1, H, W], dtype: uint8
    UINT16 = "uint16"# eg: uint16 tensor, shape: [1, H, W], dtype: uint16
    IMAGE_LUMINANCE = "image_luminance" # eg: image mode='L', dtype: uint8
    IMAGE_ALPHA = "image_alpha" # eg: image mode='RGBA', dtype: uint8, alpha channel is the mask
    IMAGE_COMPOUND = "image_compound"# eg: each channel is a different mask

class MaskField(BaseModel):
    """A masking primitive field."""

    asset_id: str = Field(description="The id/name of the mask image within the asset cache system.")
    mode: EMaskMode = Field(description="The masking mode specifies how the mask is represented.")


@invocation_output("adv_mask_output")
class AdvancedMaskOutput(BaseInvocationOutput):

    mask: MaskField = OutputField(description="The mask.")
    image: ImageField = OutputField(description="The mask image.")
    width: int = OutputField(description="The width of the mask in pixels.")
    height: int = OutputField(description="The height of the mask in pixels.")
