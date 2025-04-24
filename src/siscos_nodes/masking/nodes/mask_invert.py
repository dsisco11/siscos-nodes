
import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.shared.invocation_context import InvocationContext

from ...util.primitives import MaskingField, MaskingNodeOutput


@invocation(
    "invert_mask",
    title="Invert Mask",
    tags=["mask", "math", "invert"],
    category="mask",
    version="0.0.1",
)
class InvertMaskInvocation(BaseInvocation):
    """Inverts a mask"""

    mask: MaskingField = InputField(title="Mask")

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        mask_in = self.mask.load(context)
        mask_in = torch.sub(1.0, mask_in)
        mask_out_id = context.tensors.save(mask_in)
        return MaskingNodeOutput(
            mask=MaskingField(asset_id=mask_out_id, mode=self.mask.mode)
        )
