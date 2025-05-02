
import torch
import torch.nn.functional as F
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.shared.invocation_context import InvocationContext

from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
    MaskingNodeOutput,
)


@invocation(
    "mask_slope_select",
    title="Slope Select",
    tags=["mask", "filter", "slope select"],
    category="mask",
    version="0.0.1",
)
class MaskSlopeSelectInvocation(BaseInvocation):
    """Performs a geodesic slope floodfill operation on the input mask."""

    mask: MaskingField = InputField(title="Mask")
    threshold: float = InputField(
        title="Threshold",
        default=1.0,
    )

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        mask_in = self.mask.load(context)
        # Apply the slope floodfill reconstruction
        mask_out = self.slope_floodfill_reconstruct(mask_in.squeeze(0), self.threshold)
        return MaskingNodeOutput(
            mask=MaskingField.build(
                context=context,
                tensor=mask_out,
                mode=self.mask.mode
            ),
        )
    
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def slope_floodfill_reconstruct(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Grayscale geodesic reconstruction:
        start from seed = tensor (where >=threshold) else −∞,
        repeatedly do rec = min(max_pool(rec), tensor) until convergence.
        """
        if tensor.dim() != 2:
            raise ValueError("Input must be 2D")

        # initialize reconstruction image
        rec = tensor.clone()
        rec[rec < threshold] = float('-inf')

        # iterate until stable
        while True:
            old = rec
            # 3×3 max‐pool = dilation (8‐connected)
            dil = F.max_pool2d(rec.unsqueeze(0).unsqueeze(0),
                            kernel_size=3, stride=1, padding=1)
            dil = dil.squeeze(0).squeeze(0)
            # constrain by original heights
            rec = torch.min(dil, tensor)
            if torch.equal(rec, old):
                break

        selection = rec > float('-inf')
        return tensor.mul(selection)
