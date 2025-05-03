
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
)


@invocation(
    "mask_cast_to_image",
    title="Convert Mask To Image",
    tags=["mask"],
    category="mask",
    version="0.0.1",
)
class CastMaskToImageInvocation(BaseInvocation):
    """
    Converts a mask to an image.
    """

    mask: MaskingField = InputField(title="Mask")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        tensor = self.mask.load(context)
        # Extract width/height from tensor
        width, height = tensor.shape[-2:]
        return ImageOutput(image=ImageField(image_name=self.mask.asset_id), width=width, height=height)