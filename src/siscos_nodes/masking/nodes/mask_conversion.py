
import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.shared.invocation_context import (
    ImageCategory,
    InvocationContext,
)
from torchvision.transforms.functional import to_pil_image as tensor_to_pil

from ...util.primitives import (
    EMaskingMode,
    LMaskingMode,
    MaskingField,
    MaskingNodeOutput,
)
from ...util.tensor_common import apply_feathering_ellipse


@invocation(
    "convert_mask",
    title="Convert Mask",
    tags=["mask", "convert"],
    category="mask",
    version="0.0.1",
)
class ConvertMaskInvocation(BaseInvocation):
    """Converts a gradient mask into a bit mask."""

    mask: MaskingField = InputField(title="Mask")
    mode: LMaskingMode = InputField(title="Mode", default=EMaskingMode.GRADIENT)
    strength: float = InputField(title="Strength", default=0.25, description="Strength of the conversion.\nE.g: when converting TO a bool-mask, this is the threshold.\nWhen converting FROM a bool-mask, this is the feathering distance.")

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        tensor = self.mask.load(context)
        if (self.mode == self.mask.mode): # converting to the same mask type (no-op)
            return MaskingNodeOutput(
                mask=MaskingField(asset_id=self.mask.asset_id, mode=self.mask.mode)
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
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK)
            case EMaskingMode.IMAGE_COMPOUND:
                img = tensor_to_pil(tensor, mode='RGBA')
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK)
            case EMaskingMode.IMAGE_LUMINANCE:
                img = tensor_to_pil(tensor, mode='L')
                mask_out_id = context.images.save(img, image_category=ImageCategory.MASK)
            case EMaskingMode.BOOLEAN | EMaskingMode.GRADIENT:
                mask_out_id = context.tensors.save(tensor)

        return MaskingNodeOutput(
            mask=MaskingField(asset_id=mask_out_id, mode=self.mask.mode)
        )
