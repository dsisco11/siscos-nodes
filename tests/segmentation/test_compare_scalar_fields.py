import pytest
import torch

from siscos_nodes.src.siscos_nodes.segmentation.nodes.segmentation_node import (
    EMixingMode,
    compare_scalar_fields,
)


@pytest.mark.parametrize(
    "mode, strength, lhs, rhs, expected_value",
    [
        # ===== ADD =====
        (EMixingMode.ADD, 0, 0, 0, 0),
        (EMixingMode.ADD, 0, 0, 1, 0),
        (EMixingMode.ADD, 0, 1, 0, 1),
        (EMixingMode.ADD, 0, 1, 1, 1),
        (EMixingMode.ADD, 1, 0, 0, 0),
        (EMixingMode.ADD, 1, 1, 0, 1),
        (EMixingMode.ADD, 1, 0, 1, 1),
        (EMixingMode.ADD, 1, 1, 1, 2),
        (EMixingMode.ADD, 0.5, 0, 0, 0),
        (EMixingMode.ADD, 0.5, 0, 1, 0.5),
        (EMixingMode.ADD, 0.5, 1, 0, 1),
        (EMixingMode.ADD, 0.5, 1, 1, 1.5),
        # ===== SUBTRACT =====
        (EMixingMode.SUBTRACT, 0, 0, 0, 0),
        (EMixingMode.SUBTRACT, 0, 0, 1, 0),
        (EMixingMode.SUBTRACT, 0, 1, 0, 1),
        (EMixingMode.SUBTRACT, 0, 1, 1, 1),
        (EMixingMode.SUBTRACT, 1, 0, 0, 0),
        (EMixingMode.SUBTRACT, 1, 1, 0, 1),
        (EMixingMode.SUBTRACT, 1, 0, 1, 0),
        (EMixingMode.SUBTRACT, 1, 1, 1, 0),
        (EMixingMode.SUBTRACT, 0.5, 0, 0, 0),
        (EMixingMode.SUBTRACT, 0.5, 0, 1, 0),
        (EMixingMode.SUBTRACT, 0.5, 1, 0, 1),
        (EMixingMode.SUBTRACT, 0.5, 1, 1, 0.5),
        # ===== MULTIPLY =====
        (EMixingMode.MULTIPLY, 0, 0, 0, 0),
        (EMixingMode.MULTIPLY, 0, 0, 1, 0),
        (EMixingMode.MULTIPLY, 0, 1, 0, 0),
        (EMixingMode.MULTIPLY, 0, 1, 1, 0),
        (EMixingMode.MULTIPLY, 1, 0, 0, 0),
        (EMixingMode.MULTIPLY, 1, 1, 0, 0),
        (EMixingMode.MULTIPLY, 1, 0, 1, 0),
        (EMixingMode.MULTIPLY, 1, 1, 1, 1),
        (EMixingMode.MULTIPLY, 0.5, 0, 0, 0),
        (EMixingMode.MULTIPLY, 0.5, 0, 1, 0),
        (EMixingMode.MULTIPLY, 0.5, 1, 0, 0),
        (EMixingMode.MULTIPLY, 0.5, 1, 1, 0.5),
        # ===== SUPPRESS =====
        (EMixingMode.SUPPRESS, 0, 0, 0, 0),
        (EMixingMode.SUPPRESS, 0, 0, 1, 0),
        (EMixingMode.SUPPRESS, 0, 1, 0, 1),
        (EMixingMode.SUPPRESS, 0, 1, 1, 1),
        (EMixingMode.SUPPRESS, 1, 0, 0, 0),
        (EMixingMode.SUPPRESS, 1, 1, 0, 1),
        (EMixingMode.SUPPRESS, 1, 0, 1, 0),
        (EMixingMode.SUPPRESS, 1, 1, 1, 0),
        (EMixingMode.SUPPRESS, 0.5, 0, 0, 0),
        (EMixingMode.SUPPRESS, 0.5, 0, 1, 0),
        (EMixingMode.SUPPRESS, 0.5, 1, 0, 1),
        (EMixingMode.SUPPRESS, 0.5, 1, 1, 0.5),
        (EMixingMode.SUPPRESS, 0.2, 1, 1, 0.8),
        (EMixingMode.SUPPRESS, 0.8, 1, 1, 0.2),
        # ===== AVERAGE =====
        (EMixingMode.AVERAGE, 0, 0, 0, 0),
        (EMixingMode.AVERAGE, 0, 0, 1, 0),
        (EMixingMode.AVERAGE, 0, 1, 0, 0.5),
        (EMixingMode.AVERAGE, 0, 1, 1, 0.5),
        (EMixingMode.AVERAGE, 1, 0, 0, 0),
        (EMixingMode.AVERAGE, 1, 1, 0, 0.5),
        (EMixingMode.AVERAGE, 1, 0, 1, 0.5),
        (EMixingMode.AVERAGE, 1, 1, 1, 1),
        (EMixingMode.AVERAGE, 0.5, 0, 0, 0),
        (EMixingMode.AVERAGE, 0.5, 0, 1, 0.25),
        (EMixingMode.AVERAGE, 0.5, 1, 0, 0.5),
        (EMixingMode.AVERAGE, 0.5, 1, 1, 0.75),
        # ===== MIN =====
        (EMixingMode.MIN, 0, 0, 0, 0),
        (EMixingMode.MIN, 0, 0, 0, 0),
        (EMixingMode.MIN, 0, 1, 0, 0),
        (EMixingMode.MIN, 0, 1, 0.5, 0),
        (EMixingMode.MIN, 1, 0, 0, 0),
        (EMixingMode.MIN, 1, 1, 0, 0),
        (EMixingMode.MIN, 1, 0, 0.5, 0),
        (EMixingMode.MIN, 1, 1, 0.5, 0.5),
        # ===== MAX =====
        (EMixingMode.MAX, 0, 0, 0, 0),
        (EMixingMode.MAX, 0, 0, 0.5, 0),
        (EMixingMode.MAX, 0, 1, 0, 1),
        (EMixingMode.MAX, 0, 1, 0.5, 1),
        (EMixingMode.MAX, 1, 0, 0, 0),
        (EMixingMode.MAX, 1, 1, 0, 1),
        (EMixingMode.MAX, 1, 0, 0.5, 0.5),
        (EMixingMode.MAX, 1, 1, 0.5, 1),
        # ===== AND =====
        (EMixingMode.AND, 0, 0, 0, 0),
        (EMixingMode.AND, 0, 0, 1, 0),
        (EMixingMode.AND, 0, 1, 0, 0),
        (EMixingMode.AND, 0, 1, 1, 0),
        (EMixingMode.AND, 1, 0, 0, 0),
        (EMixingMode.AND, 1, 1, 0, 0),
        (EMixingMode.AND, 1, 0, 1, 0),
        (EMixingMode.AND, 1, 1, 1, 1),
        (EMixingMode.AND, 0.5, 0, 0, 0),
        (EMixingMode.AND, 0.5, 0, 1, 0),
        (EMixingMode.AND, 0.5, 1, 0, 0),
        (EMixingMode.AND, 0.5, 1, 1, 1),
        #
        (EMixingMode.AND, 0.49, 0, 0.5, 0),
        (EMixingMode.AND, 0.49, 1, 0.5, 0),
        (EMixingMode.AND, 0.51, 0, 0.5, 0),
        (EMixingMode.AND, 0.51, 1, 0.5, 1),
        #
        (EMixingMode.AND, 0, 0, -1, 0),
        (EMixingMode.AND, 0, -1, 0, 0),
        (EMixingMode.AND, 0, -1, 1, 0),
        (EMixingMode.AND, 1, 0, -1, 0),
        (EMixingMode.AND, 1, 1, -1, 0),
        (EMixingMode.AND, 1, -1, 1, 0),
        # ===== OR =====
        (EMixingMode.OR, 0, 0, 0, 0),
        (EMixingMode.OR, 0, 0, 1, 0),
        (EMixingMode.OR, 0, 1, 0, 1),
        (EMixingMode.OR, 0, 1, 1, 1),
        (EMixingMode.OR, 1, 0, 0, 0),
        (EMixingMode.OR, 1, 0, 1, 1),
        (EMixingMode.OR, 1, 1, 0, 1),
        (EMixingMode.OR, 1, 1, 1, 1),
        #
        (EMixingMode.OR, 0.4, 0, 0.5, 0),
        (EMixingMode.OR, 0.51, 0, 0.5, 1),
        (EMixingMode.OR, 1, 0, 1, 1),
        (EMixingMode.OR, 0.99, 0, 1, 1),
        (EMixingMode.OR, 1, 1, 1, 1),
        (EMixingMode.OR, 1, -1, 0, 0),
        (EMixingMode.OR, 1, -1, -1, 0),
        (EMixingMode.OR, 1, -1, 1, 1),
        # ===== XOR =====
        (EMixingMode.XOR, 0, 0, 0, 0),
        (EMixingMode.XOR, 0, 0, 1, 0),
        (EMixingMode.XOR, 0, 1, 0, 1),
        (EMixingMode.XOR, 0, 1, 1, 1),
        (EMixingMode.XOR, 1, 0, 0, 0),
        (EMixingMode.XOR, 1, 1, 0, 1),
        (EMixingMode.XOR, 1, 0, 1, 1),
        (EMixingMode.XOR, 1, 1, 1, 0),
        #
        (EMixingMode.XOR, 0.5, 0, 0, 0),
        (EMixingMode.XOR, 0.5, 0, 1, 1),
        (EMixingMode.XOR, 0.5, 1, 0, 1),
        (EMixingMode.XOR, 0.5, 1, 1, 0),
        #
        (EMixingMode.XOR, 0.49, 0, 0.5, 0),
        (EMixingMode.XOR, 0.49, 1, 0.5, 1),
        (EMixingMode.XOR, 0.51, 0, 0.5, 1),
        (EMixingMode.XOR, 0.51, 1, 0.5, 0),
        #
        (EMixingMode.XOR, 0, 0, -1, 0),
        (EMixingMode.XOR, 0, -1, 0, 0),
        (EMixingMode.XOR, 0, -1, 1, 0),
        (EMixingMode.XOR, 1, 0, -1, 0),
        (EMixingMode.XOR, 1, 1, -1, 1),
        (EMixingMode.XOR, 1, -1, 1, 1),
    ],
)
def test_compare_scalar_fields(mode: EMixingMode, strength: float, lhs: float, rhs: float, expected_value: float):
    """
    Test the _compare_prompts function with different modes and strengths.
    """
    # Prepare the input tensors
    tensor_1 = torch.full((1, 1, 4, 4), lhs, dtype=torch.float32)
    tensor_2 = torch.full((1, 1, 4, 4), rhs, dtype=torch.float32)
    expected = torch.full((1, 1, 4, 4), expected_value, dtype=torch.float32)

    # Call the function under test
    result = compare_scalar_fields(mode, tensor_1, tensor_2, strength)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected_value}, got {result.tolist()[0][0][0][0]}"