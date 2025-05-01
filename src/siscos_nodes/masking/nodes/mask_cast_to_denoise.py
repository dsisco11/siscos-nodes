

import torch
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    InputField,
    OutputField,
)
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
)

from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
)
from siscos_nodes.src.siscos_nodes.util.tensor_common import gaussian_blur


@invocation_output("cast_mask_to_denoise_output")
class CastMaskToDenoiseOutput(BaseInvocationOutput):
    denoise_mask: DenoiseMaskField = OutputField(
        description="Mask for denoise model run. Values of 0.0 represent the regions to be fully denoised, and 1.0 "
        + "represent the regions to be preserved."
    )

@invocation(
    "cast_mask_to_denoise",
    title="Cast To Denoising Mask",
    tags=["mask", "convert"],
    category="mask",
    version="0.0.1",
)
class CastMaskToDenoiseInvocation(BaseInvocation):
    """Converts a mask into a denoising mask by inverting and then applying any blurring requested."""

    mask: MaskingField = InputField(title="Mask")
    radius: float = InputField(title="Radius", default=4.0, description="This is the radius of the Gaussian blur applied to the denoising mask.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CastMaskToDenoiseOutput:
        tensor: torch.Tensor = self.mask.load(context)
        # Convert to denoising mask by inverting the mask
        tensor = (1.0 - tensor).clamp(min=0.0, max=1.0)
        # Apply gaussian blur to the mask tensor
        tensor = gaussian_blur(tensor, self.radius)
        # Save the tensor to the context
        tensor_id = context.tensors.save(tensor.cpu())

        return CastMaskToDenoiseOutput(
            denoise_mask=DenoiseMaskField(mask_name=tensor_id, gradient=True)
        )
