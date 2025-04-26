from typing import Final

import pytest
import torch

from siscos_nodes.src.siscos_nodes.segmentation.common import (
    EMixingMode,
    collapse_scalar_fields,
)

# region: constants
# 1x1x2x2 tensors
T22_INC_NORM: Final = torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]]]])
T22_INC_NORM_2: Final = torch.tensor([[[[0.0, 0.0], [0.3333, 1.0]]]])
T22_INC_NORM_NEG: Final = torch.tensor([[[[-1.0, -0.3333], [0.3333, 1.0]]]])
T22_INC_0: Final = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
T22_INC_1: Final = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
T22_INC_5: Final = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
T22_0: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.0)
T22_1: Final = torch.fill(torch.empty(1, 1, 2, 2), 1.0)
T22_2: Final = torch.fill(torch.empty(1, 1, 2, 2), 2.0)
T22_0_1: Final = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
T22_1_0: Final = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
T22_1_1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
T22_0_05: Final = torch.tensor([[[[0.0, 0.0], [0.5, 0.5]]]])
T22_1_05: Final = torch.tensor([[[[1.0, 1.0], [0.5, 0.5]]]])

# 1x2x2x2 tensors
T222_0_0: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
T222_1_1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]])
T222_2_2: Final = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]])
T222_0_1: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]])
T222_1_0: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
T222_1_1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]])
T222_0_05: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]]])
T222_1_05: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]])

# 4x2x2 tensors
T422_INC_NORM: Final = torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]]])
T422_INC_NORM_2: Final = torch.tensor([[[[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]]]])
T422_INC_NORM_NEG: Final = torch.tensor([[[[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]]]])
T422_0_0_0_0: Final = torch.fill(torch.empty(4, 2, 2), 0.0)
T422_1_1_1_1: Final = torch.fill(torch.empty(4, 2, 2), 1.0)
T422_2_2_2_2: Final = torch.fill(torch.empty(4, 2, 2), 2.0)
T422_INC_0: Final = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]]])
T422_INC_1: Final = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]]])
# endregion: constants



@pytest.mark.parametrize(
    "mode, threshold, input, expected",
    [
        # ===== ADD =====
        # simple test cases
        (EMixingMode.ADD, 0, T222_0_0, T22_0),
        (EMixingMode.ADD, 0, T222_0_1, T22_1),
        (EMixingMode.ADD, 0, T222_1_0, T22_1),
        (EMixingMode.ADD, 0, T222_1_1, T22_1),
        (EMixingMode.ADD, 1, T222_0_0, T22_0),
        (EMixingMode.ADD, 1, T222_1_0, T22_0),
        (EMixingMode.ADD, 1, T222_0_1, T22_0),
        (EMixingMode.ADD, 1, T222_1_1, T22_0),
        (EMixingMode.ADD, 0.5, T222_0_0, T22_0),
        (EMixingMode.ADD, 0.5, T222_0_1, T22_1),
        (EMixingMode.ADD, 0.5, T222_1_0, T22_1),
        (EMixingMode.ADD, 0.5, T222_1_1, T22_1),
        # complex test cases
        (EMixingMode.ADD, 0, T222_0_0, T22_0),
        # ===== SUBTRACT =====
        # (ECompareMode.SUBTRACT, 0, T222_0_0, T22_0),
        # (ECompareMode.SUBTRACT, 0, T222_0_1, T22_0),
        # (ECompareMode.SUBTRACT, 0, T222_1_0, T22_1),
        # (ECompareMode.SUBTRACT, 0, T222_1_1, T22_1),
        # (ECompareMode.SUBTRACT, 1, T222_0_0, T22_0),
        # (ECompareMode.SUBTRACT, 1, T222_1_0, T22_1),
        # (ECompareMode.SUBTRACT, 1, T222_0_1, -1),
        # (ECompareMode.SUBTRACT, 1, T222_1_1, T22_0),
        # (ECompareMode.SUBTRACT, 0.5, T222_0_0, T22_0),
        # (ECompareMode.SUBTRACT, 0.5, T222_0_1, -0.5),
        # (ECompareMode.SUBTRACT, 0.5, T222_1_0, T22_1),
        # (ECompareMode.SUBTRACT, 0.5, T222_1_1, 0.5),
        # ===== MULTIPLY =====
        # (ECompareMode.MULTIPLY, 0, T222_0_0, T22_0),
        # (ECompareMode.MULTIPLY, 0, T222_0_1, T22_0),
        # (ECompareMode.MULTIPLY, 0, T222_1_0, T22_0),
        # (ECompareMode.MULTIPLY, 0, T222_1_1, T22_0),
        # (ECompareMode.MULTIPLY, 1, T222_0_0, T22_0),
        # (ECompareMode.MULTIPLY, 1, T222_1_0, T22_0),
        # (ECompareMode.MULTIPLY, 1, T222_0_1, T22_0),
        # (ECompareMode.MULTIPLY, 1, T222_1_1, T22_1),
        # (ECompareMode.MULTIPLY, 0.5, T222_0_0, T22_0),
        # (ECompareMode.MULTIPLY, 0.5, T222_0_1, T22_0),
        # (ECompareMode.MULTIPLY, 0.5, T222_1_0, T22_0),
        # (ECompareMode.MULTIPLY, 0.5, T222_1_1, 0.5),
        # # ===== SUPPRESS =====
        # (ECompareMode.SUPPRESS, 0, T222_0_0, T22_0),
        # (ECompareMode.SUPPRESS, 0, T222_0_1, T22_0),
        # (ECompareMode.SUPPRESS, 0, T222_1_0, T22_1),
        # (ECompareMode.SUPPRESS, 0, T222_1_1, T22_1),
        # (ECompareMode.SUPPRESS, 1, T222_0_0, T22_0),
        # (ECompareMode.SUPPRESS, 1, T222_1_0, T22_1),
        # (ECompareMode.SUPPRESS, 1, T222_0_1, T22_0),
        # (ECompareMode.SUPPRESS, 1, T222_1_1, T22_0),
        # (ECompareMode.SUPPRESS, 0.5, T222_0_0, T22_0),
        # (ECompareMode.SUPPRESS, 0.5, T222_0_1, T22_0),
        # (ECompareMode.SUPPRESS, 0.5, T222_1_0, T22_1),
        # (ECompareMode.SUPPRESS, 0.5, T222_1_1, 0.5),
        # (ECompareMode.SUPPRESS, 0.2, T222_1_1, 0.8),
        # (ECompareMode.SUPPRESS, 0.8, T222_1_1, 0.2),
        # # ===== AVERAGE =====
        # (ECompareMode.AVERAGE, 0, T222_0_0, T22_0),
        # (ECompareMode.AVERAGE, 0, T222_0_1, T22_0),
        # (ECompareMode.AVERAGE, 0, T222_1_0, 0.5),
        # (ECompareMode.AVERAGE, 0, T222_1_1, 0.5),
        # (ECompareMode.AVERAGE, 1, T222_0_0, T22_0),
        # (ECompareMode.AVERAGE, 1, T222_1_0, 0.5),
        # (ECompareMode.AVERAGE, 1, T222_0_1, 0.5),
        # (ECompareMode.AVERAGE, 1, T222_1_1, T22_1),
        # (ECompareMode.AVERAGE, 0.5, T222_0_0, T22_0),
        # (ECompareMode.AVERAGE, 0.5, T222_0_1, 0.25),
        # (ECompareMode.AVERAGE, 0.5, T222_1_0, 0.5),
        # (ECompareMode.AVERAGE, 0.5, T222_1_1, 0.75),
        # # ===== MIN =====
        # (ECompareMode.MIN, 0, T222_0_0, T22_0),
        # (ECompareMode.MIN, 0, T222_0_0, T22_0),
        # (ECompareMode.MIN, 0, T222_1_0, T22_0),
        # (ECompareMode.MIN, 0, T222_1_05, T22_0),
        # (ECompareMode.MIN, 1, T222_0_0, T22_0),
        # (ECompareMode.MIN, 1, T222_1_0, T22_0),
        # (ECompareMode.MIN, 1, T222_0_05, T22_0),
        # (ECompareMode.MIN, 1, T222_1_05, 0.5),
        # # ===== MAX =====
        # (ECompareMode.MAX, 0, T222_0_0, T22_0),
        # (ECompareMode.MAX, 0, T222_0_05, T22_0),
        # (ECompareMode.MAX, 0, T222_1_0, T22_1),
        # (ECompareMode.MAX, 0, T222_1_05, T22_1),
        # (ECompareMode.MAX, 1, T222_0_0, T22_0),
        # (ECompareMode.MAX, 1, T222_1_0, T22_1),
        # (ECompareMode.MAX, 1, T222_0_05, 0.5),
        # (ECompareMode.MAX, 1, T222_1_05, T22_1),
        # # ===== AND =====
        # (ECompareMode.AND, 0, T222_0_0, T22_0),
        # (ECompareMode.AND, 0, T222_0_1, T22_0),
        # (ECompareMode.AND, 0, T222_1_0, T22_0),
        # (ECompareMode.AND, 0, T222_1_1, T22_0),
        # (ECompareMode.AND, 1, T222_0_0, T22_0),
        # (ECompareMode.AND, 1, T222_1_0, T22_0),
        # (ECompareMode.AND, 1, T222_0_1, T22_0),
        # (ECompareMode.AND, 1, T222_1_1, T22_1),
        # (ECompareMode.AND, 0.5, T222_0_0, T22_0),
        # (ECompareMode.AND, 0.5, T222_0_1, T22_0),
        # (ECompareMode.AND, 0.5, T222_1_0, T22_0),
        # (ECompareMode.AND, 0.5, T222_1_1, T22_1),
        # #
        # (ECompareMode.AND, 0.49, T222_0_05, T22_0),
        # (ECompareMode.AND, 0.49, T222_1_05, T22_0),
        # (ECompareMode.AND, 0.51, T222_0_05, T22_0),
        # (ECompareMode.AND, 0.51, T222_1_05, T22_1),
        # #
        # (ECompareMode.AND, 0, T222_0_-1, T22_0),
        # (ECompareMode.AND, 0, T222_-1_0, T22_0),
        # (ECompareMode.AND, 0, T222_-1_1, T22_0),
        # (ECompareMode.AND, 1, T222_0_-1, T22_0),
        # (ECompareMode.AND, 1, T222_1_-1, T22_0),
        # (ECompareMode.AND, 1, T222_-1_1, T22_0),
        # # ===== OR =====
        # (ECompareMode.OR, 0, T222_0_0, T22_0),
        # (ECompareMode.OR, 0, T222_0_1, T22_0),
        # (ECompareMode.OR, 0, T222_1_0, T22_1),
        # (ECompareMode.OR, 0, T222_1_1, T22_1),
        # (ECompareMode.OR, 1, T222_0_0, T22_0),
        # (ECompareMode.OR, 1, T222_0_1, T22_1),
        # (ECompareMode.OR, 1, T222_1_0, T22_1),
        # (ECompareMode.OR, 1, T222_1_1, T22_1),
        # #
        # (ECompareMode.OR, 0.4, T222_0_05, T22_0),
        # (ECompareMode.OR, 0.51, T222_0_05, T22_1),
        # (ECompareMode.OR, 1, T222_0_1, T22_1),
        # (ECompareMode.OR, 0.99, T222_0_1, T22_1),
        # (ECompareMode.OR, 1, T222_1_1, T22_1),
        # (ECompareMode.OR, 1, T222_-1_0, T22_0),
        # (ECompareMode.OR, 1, T222_-1_-1, T22_0),
        # (ECompareMode.OR, 1, T222_-1_1, T22_1),
        # # ===== XOR =====
        # (ECompareMode.XOR, 0, T222_0_0, T22_0),
        # (ECompareMode.XOR, 0, T222_0_1, T22_0),
        # (ECompareMode.XOR, 0, T222_1_0, T22_1),
        # (ECompareMode.XOR, 0, T222_1_1, T22_1),
        # (ECompareMode.XOR, 1, T222_0_0, T22_0),
        # (ECompareMode.XOR, 1, T222_1_0, T22_1),
        # (ECompareMode.XOR, 1, T222_0_1, T22_1),
        # (ECompareMode.XOR, 1, T222_1_1, T22_0),
        # #
        # (ECompareMode.XOR, 0.5, T222_0_0, T22_0),
        # (ECompareMode.XOR, 0.5, T222_0_1, T22_1),
        # (ECompareMode.XOR, 0.5, T222_1_0, T22_1),
        # (ECompareMode.XOR, 0.5, T222_1_1, T22_0),
        # #
        # (ECompareMode.XOR, 0.49, T222_0_05, T22_0),
        # (ECompareMode.XOR, 0.49, T222_1_05, T22_1),
        # (ECompareMode.XOR, 0.51, T222_0_05, T22_1),
        # (ECompareMode.XOR, 0.51, T222_1_05, T22_0),
        # #
        # (ECompareMode.XOR, 0, T222_0_-1, T22_0),
        # (ECompareMode.XOR, 0, T222_-1_0, T22_0),
        # (ECompareMode.XOR, 0, T222_-1_1, T22_0),
        # (ECompareMode.XOR, 1, T222_0_-1, T22_0),
        # (ECompareMode.XOR, 1, T222_1_-1, T22_1),
        # (ECompareMode.XOR, 1, T222_-1_1, T22_1),
    ],
)
def test_collapse_scalar_fields(mode: EMixingMode, threshold: float, input: torch.Tensor, expected: torch.Tensor):
    """
    Test the collapse_gradient_masks function with different modes and strengths.
    """
    # Call the function under test
    result = collapse_scalar_fields(input, threshold, mode)

    # Check the result
    # assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"""Received: {result}
    Expected: {expected}"""