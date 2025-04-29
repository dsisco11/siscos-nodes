
from ast import TypeAlias
from typing import Any, Tuple, Union

import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, TensorField, UIType
from invokeai.app.services.shared.invocation_context import (
    ImageCategory,
    InvocationContext,
)
from torchvision.transforms.functional import to_pil_image as tensor_to_pil
from torchvision.transforms.functional import to_tensor as pil_to_tensor

from ...util.primitives import (
    EMaskingMode,
    LMaskingMode,
    MaskingField,
    MaskingNodeOutput,
)
from ...util.tensor_common import apply_feathering_ellipse

MaskLike = MaskingField | TensorField | ImageField

@invocation(
    "convert_mask",
    title="Convert Mask",
    tags=["mask", "convert"],
    category="mask",
    version="0.1.1",
)
class ConvertMaskInvocation(BaseInvocation):
    """Converts a gradient mask into a bit mask."""

    mask: MaskLike = InputField(title="Mask", ui_type=UIType.Any)
    mode: LMaskingMode = InputField(title="Mode", default=EMaskingMode.GRADIENT)
    strength: float = InputField(title="Strength", default=0.25, description="Strength of the conversion.\nE.g: when converting TO a bool-mask, this is the threshold.\nWhen converting FROM a bool-mask, this is the feathering distance.")

    def loadMaskTensor(self, context: InvocationContext) -> Tuple[EMaskingMode, torch.Tensor]:
        """Resolve the mask tensor from the input field."""
        if isinstance(self.mask, MaskingField):
            return self.mask.mode, self.mask.load(context)
        elif isinstance(self.mask, TensorField):
            return EMaskingMode.BOOLEAN, context.tensors.load(self.mask.tensor_name)
        elif isinstance(self.mask, ImageField):
            return EMaskingMode.IMAGE_LUMINANCE, pil_to_tensor(context.images.get_pil(self.mask.image_name, mode='L'))
        else:
            raise ValueError(f"Unsupported mask type: {type(self.mask)}")
        
    def getMaskAssetId(self) -> str:
        """Resolve the mask asset ID from the input field."""
        if isinstance(self.mask, MaskingField):
            return self.mask.asset_id
        elif isinstance(self.mask, TensorField):
            return self.mask.tensor_name
        elif isinstance(self.mask, ImageField):
            return self.mask.image_name
        else:
            raise ValueError(f"Unsupported mask type: {type(self.mask)}")

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        tpl = self.loadMaskTensor(context)
        originalMode:EMaskingMode = tpl[0]
        tensor: torch.Tensor = tpl[1]

        if (self.mode == originalMode): # converting to the same mask type (no-op)
            return MaskingNodeOutput(
                mask=MaskingField(asset_id=self.getMaskAssetId(), mode=originalMode)
            )

        # Figure out if we are converting to or from a boolean mask so we can apply the strength correctly.
        if (self.mode == EMaskingMode.BOOLEAN):
            # Converting TO a boolean mask.
            tensor = tensor.sub(self.strength).to(torch.bool)
        else:
            # Converting FROM a boolean mask.
            tensor = apply_feathering_ellipse(tensor.to(torch.float32), self.strength)

        mask_out_id: str = None
        match (self.mode):
            case EMaskingMode.IMAGE_ALPHA:
                img = tensor_to_pil(tensor, mode='RGBA')
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK).image_name
            case EMaskingMode.IMAGE_COMPOUND:
                img = tensor_to_pil(tensor, mode='RGBA')
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK).image_name
            case EMaskingMode.IMAGE_LUMINANCE:
                img = tensor_to_pil(tensor, mode='L')
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK).image_name
            case EMaskingMode.BOOLEAN | EMaskingMode.GRADIENT:
                mask_out_id = context.tensors.save(tensor)

        return MaskingNodeOutput(
            mask=MaskingField(asset_id=mask_out_id, mode=originalMode)
        )
