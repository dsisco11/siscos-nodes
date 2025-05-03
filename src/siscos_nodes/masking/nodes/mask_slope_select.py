
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
        selection = self.slope_floodfill(mask_in.squeeze(0), self.threshold)
        mask_out = mask_in.multiply(selection)
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
    def slope_floodfill(tensor: torch.Tensor, threshold: float, max_iters: int = 1000):
        """
        Starts from all locations >= threshold and spreads from each pixel to neighbors of lower-or-equal height.

        Args:
        tensor    (H×W float): your height map
        threshold   (float):   start from all locations >= threshold
        max_iters    (int):    safety cap on iterations

        Returns:
        mask (H×W bool): True wherever reachable by only descending or flat moves
        """
        if tensor.dim() != 2:
            raise ValueError("Input must be 2D")

        device = tensor.device

        # Initialize filled mask from threshold
        filled = (tensor >= threshold).float()
        prev_filled = torch.zeros_like(filled)

        # 8-connectivity kernel
        kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        for _ in range(max_iters):
            # Get neighbors' maximum height where filled
            filled_heights = filled * tensor
            neighbor_max = F.conv2d(
                filled_heights.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=1
            ).squeeze(0).squeeze(0)

            # Count filled neighbors (for positions where no neighbor is filled, prevent fill)
            neighbor_count = F.conv2d(
                filled.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=1
            ).squeeze(0).squeeze(0)

            # Candidate pixels: not yet filled, have at least one filled neighbor,
            # and have height <= neighbor_max
            candidates = (filled == 0) & (neighbor_count > 0) & (tensor <= neighbor_max)

            if not candidates.any() or torch.equal(filled, prev_filled):
                break

            prev_filled = filled.clone()
            filled[candidates] = 1.0

        return filled.to(torch.bool)
