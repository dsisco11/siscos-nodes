from enum import Enum
from typing import Literal

import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.shared.invocation_context import InvocationContext

from siscos_nodes.src.siscos_nodes.masking.enums import (
    EMaskingMode,
    EMathOperators,
    LMathOperators,
)
from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
    MaskingNodeOutput,
)
from siscos_nodes.src.siscos_nodes.util.tensor_common import resize_tensor


@invocation(
    "mask_math",
    title="Mask Math",
    tags=["mask", "math"],
    category="mask",
    version="0.0.2",
)
class MaskMathOperationInvocation(BaseInvocation):
    """Perform a mathematical operation on a mask."""

    mask_a: MaskingField = InputField(title="Mask A")
    mask_b: MaskingField = InputField(title="Mask B")
    operation: LMathOperators = InputField(title="Operation", default="subtract")

    def invoke(self, context: InvocationContext) -> MaskingNodeOutput:
        lhs = self.mask_a.load(context)
        rhs = self.mask_b.load(context)

        # ensure the tensors are the same size
        if (lhs.shape[-2:] != rhs.shape[-2:]):
            rhs = resize_tensor(rhs, target_size=(lhs.shape[-2], lhs.shape[-1]))

        if lhs.shape != rhs.shape:
            raise ValueError(f"Mask shapes do not match: {lhs.shape} != {rhs.shape}")

        result: torch.Tensor
        match (self.operation):
            case EMathOperators.ADD:
                result = torch.add(lhs, rhs)
            case EMathOperators.SUBTRACT:
                result = torch.sub(lhs, rhs)
            case EMathOperators.MULTIPLY:
                result = torch.mul(lhs, rhs)
            case EMathOperators.DIVIDE:
                result = torch.div(lhs, rhs)
            case EMathOperators.AVERAGE:
                result = torch.mean(torch.stack([lhs, rhs]), dim=0)
            case EMathOperators.MEDIAN:
                result = torch.quantile(torch.stack([lhs, rhs]), dim=0, q=0.5)
            case EMathOperators.MAX:
                result = torch.max(lhs, rhs)
            case EMathOperators.MIN:
                result = torch.min(lhs, rhs)
            case EMathOperators.XOR:
                result = torch.logical_xor(lhs, rhs)
            case EMathOperators.OR:
                result = torch.logical_or(lhs, rhs)
            case EMathOperators.AND:
                result = torch.logical_and(lhs, rhs)
            case _:
                raise ValueError(f"Unsupported operation: {self.operation}")

        result_mode: EMaskingMode = self.mask_a.mode
        return MaskingNodeOutput(
            mask=MaskingField.build(
                context=context,
                tensor=result,
                mode=result_mode,
            ),
        )
