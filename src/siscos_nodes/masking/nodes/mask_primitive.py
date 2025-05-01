from typing import List

import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.shared.invocation_context import InvocationContext

from siscos_nodes.src.siscos_nodes.masking.enums import (
    EMaskingMode,
    EMaskShape,
    LMaskShape,
)
from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
    MaskingNodeOutput,
)


@invocation(
    "mask_primitive",
    title="Mask Primitve",
    tags=["mask", "math", "primitive", "shape"],
    category="mask",
    version="0.0.1",
)
class MaskPrimitiveInvocation(BaseInvocation):
    """Construct a mask from a primitive shape."""

    shape: LMaskShape = InputField(title="Shape", default=EMaskShape.SOLID, 
        ui_choice_labels={
            EMaskShape.SOLID: "Solid",
            EMaskShape.GRADIENT_RADIAL: "Radial Gradient",
            EMaskShape.GRADIENT_VERTICAL: "Vertical Gradient",
            EMaskShape.GRADIENT_HORIZONTAL: "Horizontal Gradient",
        }
    )
    value_start: float = InputField(title="Start Value", default=1.0)
    value_end: float = InputField(title="End Value", default=0.0)
    gradient_start: float = InputField(title="Start Offset", default=0.0)
    gradient_end: float = InputField(title="End Offset", default=1.0)
    width: int = InputField(title="Width", default=512)
    height: int = InputField(title="Height", default=512)

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        # Create a mask based on the specified shape
        result = MaskPrimitiveInvocation._create_mask(
            shape=self.shape,
            value_start=self.value_start,
            value_end=self.value_end,
            gradient_start=self.gradient_start,
            gradient_end=self.gradient_end,
            width=self.width,
            height=self.height,
        )

        return MaskingNodeOutput(
            mask=MaskingField.build(
                context=context,
                tensor=result,
                mode=EMaskingMode.GRADIENT,
            ),
        )
    
    @staticmethod
    def _create_mask(shape: EMaskShape, value_start: float, value_end: float, gradient_start: float, gradient_end: float, width: int, height: int) -> torch.Tensor:
        """Create a mask based on the specified shape."""
        match (shape):
            case EMaskShape.SOLID:
                return torch.full((1, 1, height, width), value_start)
            case EMaskShape.GRADIENT_RADIAL:
                return MaskPrimitiveInvocation._create_radial_multi_gradient([value_start, value_end], [gradient_start, gradient_end], width, height)
            case EMaskShape.GRADIENT_VERTICAL:
                return MaskPrimitiveInvocation._create_vertical_multi_gradient([value_start, value_end], [gradient_start, gradient_end], width, height)
            case EMaskShape.GRADIENT_HORIZONTAL:
                return MaskPrimitiveInvocation._create_horizontal_multi_gradient([value_start, value_end], [gradient_start, gradient_end], width, height)
        raise ValueError(f"Unsupported mask shape: {shape}")

    
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _create_vertical_multi_gradient(
        values: List[float],
        offsets: List[float],
        width: int,
        height: int
    ) -> torch.Tensor:
        # Fallback if mis-configured
        if len(values) < 2 or len(offsets) != len(values):
            return torch.full((height, width), values[0] if len(values) else 0.0)

        # 1) Build a 1D ramp [0…1] down the rows
        ys = torch.arange(height, dtype=torch.float32)
        if height > 1:
            ys = ys / (height - 1)
        else:
            ys = ys * 0.0

        # 2) Pack stops into tensors
        offs = torch.tensor(offsets, dtype=torch.float32)
        vals = torch.tensor(values, dtype=torch.float32)

        # 3) Which segment each row falls into?
        bins = torch.bucketize(ys, offs)                     # [H]
        bins = torch.clamp(bins, 1, offs.size(0) - 1)

        # 4) Gather endpoint offsets & values
        o0 = offs[bins - 1]      # left edge (upper) of segment
        o1 = offs[bins]          # right edge (lower)
        v0 = vals[bins - 1]
        v1 = vals[bins]

        # 5) Interpolate each row
        alpha = (ys - o0) / (o1 - o0)                        # [H]
        grad1d = v0 + (v1 - v0) * alpha                      # [H]

        # 6) Broadcast to [H,W]
        return grad1d.unsqueeze(1).expand(height, width)


    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _create_radial_multi_gradient(
        values: List[float],
        offsets: List[float],
        width: int,
        height: int
    ) -> torch.Tensor:
        # Fallback if mis-configured
        if len(values) < 2 or len(offsets) != len(values):
            return torch.full((height, width), values[0] if len(values) else 0.0)

        # 1) Build normalized coords in [−1…+1]
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

        if width > 1:
            xs = (xs - cx) / cx
        else:
            xs = xs * 0.0

        if height > 1:
            ys = (ys - cy) / cy
        else:
            ys = ys * 0.0

        # 2) Compute per-pixel radial distance ∈ [0,√2], then clamp to [0,1]
        xx = xs * xs                  # [W]
        yy = ys * ys                  # [H]
        dist = torch.sqrt(xx.unsqueeze(0) + yy.unsqueeze(1))  # [H,W]
        dist = torch.clamp(dist, 0.0, 1.0)

        # 3) Pack stops into tensors
        offs = torch.tensor(offsets, dtype=torch.float32)
        vals = torch.tensor(values, dtype=torch.float32)

        # 4) Which segment each pixel’s radius falls into?
        bins = torch.bucketize(dist, offs)                    # [H,W]
        bins = torch.clamp(bins, 1, offs.size(0) - 1)

        # 5) Gather endpoint offsets & values
        o0 = offs[bins - 1]
        o1 = offs[bins]
        v0 = vals[bins - 1]
        v1 = vals[bins]

        # 6) Interpolate per-pixel
        alpha = (dist - o0) / (o1 - o0)
        return v0 + (v1 - v0) * alpha                        # [H,W]
    
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _create_horizontal_multi_gradient(
        values: List[float],
        offsets: List[float],
        width: int,
        height: int
    ) -> torch.Tensor:
        # ---- 1) sanity checks & Tensors ----
        # must have at least two stops
        if len(values) < 2 or len(values) != len(offsets):
            # fall back to a constant map if mis-configured
            return torch.full((height, width), values[0] if values else 0.0)

        # create 1D ramp [0…1]
        xs = torch.arange(width, dtype=torch.float32)
        if width > 1:
            xs = xs / (width - 1)
        else:
            xs = xs * 0.0

        # pack lists into tensors
        offs = torch.tensor(offsets, dtype=torch.float32)
        vals = torch.tensor(values, dtype=torch.float32)

        # ---- 2) find which segment each x falls into ----
        # bucketize: returns for each x an index i in [0..len(offs)]
        # such that offs[i-1] < x ≤ offs[i], with offs[-1] = -∞, offs[n] = +∞
        bins = torch.bucketize(xs, offs)

        # clamp to [1, n-1] so we can always do (i-1, i)
        bins = torch.clamp(bins, 1, offs.size(0) - 1)

        # ---- 3) gather segment endpoints ----
        o0 = offs[bins - 1]       # left edge of each segment
        o1 = offs[bins]           # right edge
        v0 = vals[bins - 1]       # value at left edge
        v1 = vals[bins]           # value at right edge

        # ---- 4) linear interpolate within segment ----
        alpha = (xs - o0) / (o1 - o0)   # in [0,1]
        grad1d = v0 + (v1 - v0) * alpha

        # ---- 5) broadcast to full [H,W] ----
        return grad1d.unsqueeze(0).expand(height, width)