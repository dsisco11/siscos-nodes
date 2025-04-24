import pytest
import torch
from src.siscos_nodes.segmentation.nodes.segmentation_node import (
    ECompareMode,
    _compare_prompts,
)


@pytest.mark.parametrize(
    "mode, strength, lhs, rhs, expected_value",
    [
        # ===== ADD =====
        (ECompareMode.ADD, 0, 0, 0, 0),
        (ECompareMode.ADD, 0, 0, 1, 0),
        (ECompareMode.ADD, 0, 1, 0, 1),
        (ECompareMode.ADD, 0, 1, 1, 1),
        (ECompareMode.ADD, 1, 0, 0, 0),
        (ECompareMode.ADD, 1, 1, 0, 1),
        (ECompareMode.ADD, 1, 0, 1, 1),
        (ECompareMode.ADD, 1, 1, 1, 2),
        (ECompareMode.ADD, 0.5, 0, 0, 0),
        (ECompareMode.ADD, 0.5, 0, 1, 0.5),
        (ECompareMode.ADD, 0.5, 1, 0, 1),
        (ECompareMode.ADD, 0.5, 1, 1, 1.5),
        # ===== SUBTRACT =====
        (ECompareMode.SUBTRACT, 0, 0, 0, 0),
        (ECompareMode.SUBTRACT, 0, 0, 1, 0),
        (ECompareMode.SUBTRACT, 0, 1, 0, 1),
        (ECompareMode.SUBTRACT, 0, 1, 1, 1),
        (ECompareMode.SUBTRACT, 1, 0, 0, 0),
        (ECompareMode.SUBTRACT, 1, 1, 0, 1),
        (ECompareMode.SUBTRACT, 1, 0, 1, -1),
        (ECompareMode.SUBTRACT, 1, 1, 1, 0),
        (ECompareMode.SUBTRACT, 0.5, 0, 0, 0),
        (ECompareMode.SUBTRACT, 0.5, 0, 1, -0.5),
        (ECompareMode.SUBTRACT, 0.5, 1, 0, 1),
        (ECompareMode.SUBTRACT, 0.5, 1, 1, 0.5),
        # ===== MULTIPLY =====
        (ECompareMode.MULTIPLY, 0, 0, 0, 0),
        (ECompareMode.MULTIPLY, 0, 0, 1, 0),
        (ECompareMode.MULTIPLY, 0, 1, 0, 0),
        (ECompareMode.MULTIPLY, 0, 1, 1, 0),
        (ECompareMode.MULTIPLY, 1, 0, 0, 0),
        (ECompareMode.MULTIPLY, 1, 1, 0, 0),
        (ECompareMode.MULTIPLY, 1, 0, 1, 0),
        (ECompareMode.MULTIPLY, 1, 1, 1, 1),
        (ECompareMode.MULTIPLY, 0.5, 0, 0, 0),
        (ECompareMode.MULTIPLY, 0.5, 0, 1, 0),
        (ECompareMode.MULTIPLY, 0.5, 1, 0, 0),
        (ECompareMode.MULTIPLY, 0.5, 1, 1, 0.5),
        # ===== SUPPRESS =====
        (ECompareMode.SUPPRESS, 0, 0, 0, 0),
        (ECompareMode.SUPPRESS, 0, 0, 1, 0),
        (ECompareMode.SUPPRESS, 0, 1, 0, 1),
        (ECompareMode.SUPPRESS, 0, 1, 1, 1),
        (ECompareMode.SUPPRESS, 1, 0, 0, 0),
        (ECompareMode.SUPPRESS, 1, 1, 0, 1),
        (ECompareMode.SUPPRESS, 1, 0, 1, 0),
        (ECompareMode.SUPPRESS, 1, 1, 1, 0),
        (ECompareMode.SUPPRESS, 0.5, 0, 0, 0),
        (ECompareMode.SUPPRESS, 0.5, 0, 1, 0),
        (ECompareMode.SUPPRESS, 0.5, 1, 0, 1),
        (ECompareMode.SUPPRESS, 0.5, 1, 1, 0.5),
        (ECompareMode.SUPPRESS, 0.2, 1, 1, 0.8),
        (ECompareMode.SUPPRESS, 0.8, 1, 1, 0.2),
        # ===== AVERAGE =====
        (ECompareMode.AVERAGE, 0, 0, 0, 0),
        (ECompareMode.AVERAGE, 0, 0, 1, 0),
        (ECompareMode.AVERAGE, 0, 1, 0, 0.5),
        (ECompareMode.AVERAGE, 0, 1, 1, 0.5),
        (ECompareMode.AVERAGE, 1, 0, 0, 0),
        (ECompareMode.AVERAGE, 1, 1, 0, 0.5),
        (ECompareMode.AVERAGE, 1, 0, 1, 0.5),
        (ECompareMode.AVERAGE, 1, 1, 1, 1),
        (ECompareMode.AVERAGE, 0.5, 0, 0, 0),
        (ECompareMode.AVERAGE, 0.5, 0, 1, 0.25),
        (ECompareMode.AVERAGE, 0.5, 1, 0, 0.5),
        (ECompareMode.AVERAGE, 0.5, 1, 1, 0.75),
        # ===== MIN =====
        (ECompareMode.MIN, 0, 0, 0, 0),
        (ECompareMode.MIN, 0, 0, 0, 0),
        (ECompareMode.MIN, 0, 1, 0, 0),
        (ECompareMode.MIN, 0, 1, 0.5, 0),
        (ECompareMode.MIN, 1, 0, 0, 0),
        (ECompareMode.MIN, 1, 1, 0, 0),
        (ECompareMode.MIN, 1, 0, 0.5, 0),
        (ECompareMode.MIN, 1, 1, 0.5, 0.5),
        # ===== MAX =====
        (ECompareMode.MAX, 0, 0, 0, 0),
        (ECompareMode.MAX, 0, 0, 0.5, 0),
        (ECompareMode.MAX, 0, 1, 0, 1),
        (ECompareMode.MAX, 0, 1, 0.5, 1),
        (ECompareMode.MAX, 1, 0, 0, 0),
        (ECompareMode.MAX, 1, 1, 0, 1),
        (ECompareMode.MAX, 1, 0, 0.5, 0.5),
        (ECompareMode.MAX, 1, 1, 0.5, 1),
        # ===== AND =====
        (ECompareMode.AND, 0, 0, 0, 0),
        (ECompareMode.AND, 0, 0, 1, 0),
        (ECompareMode.AND, 0, 1, 0, 0),
        (ECompareMode.AND, 0, 1, 1, 0),
        (ECompareMode.AND, 1, 0, 0, 0),
        (ECompareMode.AND, 1, 1, 0, 0),
        (ECompareMode.AND, 1, 0, 1, 0),
        (ECompareMode.AND, 1, 1, 1, 1),
        (ECompareMode.AND, 0.5, 0, 0, 0),
        (ECompareMode.AND, 0.5, 0, 1, 0),
        (ECompareMode.AND, 0.5, 1, 0, 0),
        (ECompareMode.AND, 0.5, 1, 1, 1),
        #
        (ECompareMode.AND, 0.49, 0, 0.5, 0),
        (ECompareMode.AND, 0.49, 1, 0.5, 0),
        (ECompareMode.AND, 0.51, 0, 0.5, 0),
        (ECompareMode.AND, 0.51, 1, 0.5, 1),
        #
        (ECompareMode.AND, 0, 0, -1, 0),
        (ECompareMode.AND, 0, -1, 0, 0),
        (ECompareMode.AND, 0, -1, 1, 0),
        (ECompareMode.AND, 1, 0, -1, 0),
        (ECompareMode.AND, 1, 1, -1, 0),
        (ECompareMode.AND, 1, -1, 1, 0),
        # ===== OR =====
        (ECompareMode.OR, 0, 0, 0, 0),
        (ECompareMode.OR, 0, 0, 1, 0),
        (ECompareMode.OR, 0, 1, 0, 1),
        (ECompareMode.OR, 0, 1, 1, 1),
        (ECompareMode.OR, 1, 0, 0, 0),
        (ECompareMode.OR, 1, 0, 1, 1),
        (ECompareMode.OR, 1, 1, 0, 1),
        (ECompareMode.OR, 1, 1, 1, 1),
        #
        (ECompareMode.OR, 0.4, 0, 0.5, 0),
        (ECompareMode.OR, 0.51, 0, 0.5, 1),
        (ECompareMode.OR, 1, 0, 1, 1),
        (ECompareMode.OR, 0.99, 0, 1, 1),
        (ECompareMode.OR, 1, 1, 1, 1),
        (ECompareMode.OR, 1, -1, 0, 0),
        (ECompareMode.OR, 1, -1, -1, 0),
        (ECompareMode.OR, 1, -1, 1, 1),
        # ===== XOR =====
        (ECompareMode.XOR, 0, 0, 0, 0),
        (ECompareMode.XOR, 0, 0, 1, 0),
        (ECompareMode.XOR, 0, 1, 0, 1),
        (ECompareMode.XOR, 0, 1, 1, 1),
        (ECompareMode.XOR, 1, 0, 0, 0),
        (ECompareMode.XOR, 1, 1, 0, 1),
        (ECompareMode.XOR, 1, 0, 1, 1),
        (ECompareMode.XOR, 1, 1, 1, 0),
        #
        (ECompareMode.XOR, 0.5, 0, 0, 0),
        (ECompareMode.XOR, 0.5, 0, 1, 1),
        (ECompareMode.XOR, 0.5, 1, 0, 1),
        (ECompareMode.XOR, 0.5, 1, 1, 0),
        #
        (ECompareMode.XOR, 0.49, 0, 0.5, 0),
        (ECompareMode.XOR, 0.49, 1, 0.5, 1),
        (ECompareMode.XOR, 0.51, 0, 0.5, 1),
        (ECompareMode.XOR, 0.51, 1, 0.5, 0),
        #
        (ECompareMode.XOR, 0, 0, -1, 0),
        (ECompareMode.XOR, 0, -1, 0, 0),
        (ECompareMode.XOR, 0, -1, 1, 0),
        (ECompareMode.XOR, 1, 0, -1, 0),
        (ECompareMode.XOR, 1, 1, -1, 1),
        (ECompareMode.XOR, 1, -1, 1, 1),
    ],
)
def test_compare_prompts(mode: ECompareMode, strength: float, lhs: float, rhs: float, expected_value: float):
    """
    Test the _compare_prompts function with different modes and strengths.
    """
    # Prepare the input tensors
    tensor_1 = torch.full((1, 1, 4, 4), lhs, dtype=torch.float32)
    tensor_2 = torch.full((1, 1, 4, 4), rhs, dtype=torch.float32)
    expected = torch.full((1, 1, 4, 4), expected_value, dtype=torch.float32)

    # Call the function under test
    result = _compare_prompts(mode, tensor_1, tensor_2, strength)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected_value}, got {result.tolist()[0][0][0][0]}"