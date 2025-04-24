import pytest
import torch
from src.siscos_nodes.segmentation.nodes.segmentation_node import (
    ECompareMode,
    _collapse_prompts,
)


@pytest.mark.parametrize(
    "mode, threshold, layer_values, expected_value",
    [
        # ===== ADD =====
        (ECompareMode.ADD, 0, [0, 0], 0),
        (ECompareMode.ADD, 0, [0, 1], 0),
        (ECompareMode.ADD, 0, [1, 0], 1),
        (ECompareMode.ADD, 0, [1, 1], 1),
        (ECompareMode.ADD, 1, [0, 0], 0),
        (ECompareMode.ADD, 1, [1, 0], 1),
        (ECompareMode.ADD, 1, [0, 1], 1),
        (ECompareMode.ADD, 1, [1, 1], 2),
        (ECompareMode.ADD, 0.5, [0, 0], 0),
        (ECompareMode.ADD, 0.5, [0, 1], 0.5),
        (ECompareMode.ADD, 0.5, [1, 0], 1),
        (ECompareMode.ADD, 0.5, [1, 1], 1.5),
        # ===== SUBTRACT =====
        # (ECompareMode.SUBTRACT, 0, [0, 0], 0),
        # (ECompareMode.SUBTRACT, 0, [0, 1], 0),
        # (ECompareMode.SUBTRACT, 0, [1, 0], 1),
        # (ECompareMode.SUBTRACT, 0, [1, 1], 1),
        # (ECompareMode.SUBTRACT, 1, [0, 0], 0),
        # (ECompareMode.SUBTRACT, 1, [1, 0], 1),
        # (ECompareMode.SUBTRACT, 1, [0, 1], -1),
        # (ECompareMode.SUBTRACT, 1, [1, 1], 0),
        # (ECompareMode.SUBTRACT, 0.5, [0, 0], 0),
        # (ECompareMode.SUBTRACT, 0.5, [0, 1], -0.5),
        # (ECompareMode.SUBTRACT, 0.5, [1, 0], 1),
        # (ECompareMode.SUBTRACT, 0.5, [1, 1], 0.5),
        # ===== MULTIPLY =====
        # (ECompareMode.MULTIPLY, 0, [0, 0], 0),
        # (ECompareMode.MULTIPLY, 0, [0, 1], 0),
        # (ECompareMode.MULTIPLY, 0, [1, 0], 0),
        # (ECompareMode.MULTIPLY, 0, [1, 1], 0),
        # (ECompareMode.MULTIPLY, 1, [0, 0], 0),
        # (ECompareMode.MULTIPLY, 1, [1, 0], 0),
        # (ECompareMode.MULTIPLY, 1, [0, 1], 0),
        # (ECompareMode.MULTIPLY, 1, [1, 1], 1),
        # (ECompareMode.MULTIPLY, 0.5, [0, 0], 0),
        # (ECompareMode.MULTIPLY, 0.5, [0, 1], 0),
        # (ECompareMode.MULTIPLY, 0.5, [1, 0], 0),
        # (ECompareMode.MULTIPLY, 0.5, [1, 1], 0.5),
        # # ===== SUPPRESS =====
        # (ECompareMode.SUPPRESS, 0, [0, 0], 0),
        # (ECompareMode.SUPPRESS, 0, [0, 1], 0),
        # (ECompareMode.SUPPRESS, 0, [1, 0], 1),
        # (ECompareMode.SUPPRESS, 0, [1, 1], 1),
        # (ECompareMode.SUPPRESS, 1, [0, 0], 0),
        # (ECompareMode.SUPPRESS, 1, [1, 0], 1),
        # (ECompareMode.SUPPRESS, 1, [0, 1], 0),
        # (ECompareMode.SUPPRESS, 1, [1, 1], 0),
        # (ECompareMode.SUPPRESS, 0.5, [0, 0], 0),
        # (ECompareMode.SUPPRESS, 0.5, [0, 1], 0),
        # (ECompareMode.SUPPRESS, 0.5, [1, 0], 1),
        # (ECompareMode.SUPPRESS, 0.5, [1, 1], 0.5),
        # (ECompareMode.SUPPRESS, 0.2, [1, 1], 0.8),
        # (ECompareMode.SUPPRESS, 0.8, [1, 1], 0.2),
        # # ===== AVERAGE =====
        # (ECompareMode.AVERAGE, 0, [0, 0], 0),
        # (ECompareMode.AVERAGE, 0, [0, 1], 0),
        # (ECompareMode.AVERAGE, 0, [1, 0], 0.5),
        # (ECompareMode.AVERAGE, 0, [1, 1], 0.5),
        # (ECompareMode.AVERAGE, 1, [0, 0], 0),
        # (ECompareMode.AVERAGE, 1, [1, 0], 0.5),
        # (ECompareMode.AVERAGE, 1, [0, 1], 0.5),
        # (ECompareMode.AVERAGE, 1, [1, 1], 1),
        # (ECompareMode.AVERAGE, 0.5, [0, 0], 0),
        # (ECompareMode.AVERAGE, 0.5, [0, 1], 0.25),
        # (ECompareMode.AVERAGE, 0.5, [1, 0], 0.5),
        # (ECompareMode.AVERAGE, 0.5, [1, 1], 0.75),
        # # ===== MIN =====
        # (ECompareMode.MIN, 0, [0, 0], 0),
        # (ECompareMode.MIN, 0, [0, 0], 0),
        # (ECompareMode.MIN, 0, [1, 0], 0),
        # (ECompareMode.MIN, 0, [1, 0.5], 0),
        # (ECompareMode.MIN, 1, [0, 0], 0),
        # (ECompareMode.MIN, 1, [1, 0], 0),
        # (ECompareMode.MIN, 1, [0, 0.5], 0),
        # (ECompareMode.MIN, 1, [1, 0.5], 0.5),
        # # ===== MAX =====
        # (ECompareMode.MAX, 0, [0, 0], 0),
        # (ECompareMode.MAX, 0, [0, 0.5], 0),
        # (ECompareMode.MAX, 0, [1, 0], 1),
        # (ECompareMode.MAX, 0, [1, 0.5], 1),
        # (ECompareMode.MAX, 1, [0, 0], 0),
        # (ECompareMode.MAX, 1, [1, 0], 1),
        # (ECompareMode.MAX, 1, [0, 0.5], 0.5),
        # (ECompareMode.MAX, 1, [1, 0.5], 1),
        # # ===== AND =====
        # (ECompareMode.AND, 0, [0, 0], 0),
        # (ECompareMode.AND, 0, [0, 1], 0),
        # (ECompareMode.AND, 0, [1, 0], 0),
        # (ECompareMode.AND, 0, [1, 1], 0),
        # (ECompareMode.AND, 1, [0, 0], 0),
        # (ECompareMode.AND, 1, [1, 0], 0),
        # (ECompareMode.AND, 1, [0, 1], 0),
        # (ECompareMode.AND, 1, [1, 1], 1),
        # (ECompareMode.AND, 0.5, [0, 0], 0),
        # (ECompareMode.AND, 0.5, [0, 1], 0),
        # (ECompareMode.AND, 0.5, [1, 0], 0),
        # (ECompareMode.AND, 0.5, [1, 1], 1),
        # #
        # (ECompareMode.AND, 0.49, [0, 0.5], 0),
        # (ECompareMode.AND, 0.49, [1, 0.5], 0),
        # (ECompareMode.AND, 0.51, [0, 0.5], 0),
        # (ECompareMode.AND, 0.51, [1, 0.5], 1),
        # #
        # (ECompareMode.AND, 0, [0, -1], 0),
        # (ECompareMode.AND, 0, [-1, 0], 0),
        # (ECompareMode.AND, 0, [-1, 1], 0),
        # (ECompareMode.AND, 1, [0, -1], 0),
        # (ECompareMode.AND, 1, [1, -1], 0),
        # (ECompareMode.AND, 1, [-1, 1], 0),
        # # ===== OR =====
        # (ECompareMode.OR, 0, [0, 0], 0),
        # (ECompareMode.OR, 0, [0, 1], 0),
        # (ECompareMode.OR, 0, [1, 0], 1),
        # (ECompareMode.OR, 0, [1, 1], 1),
        # (ECompareMode.OR, 1, [0, 0], 0),
        # (ECompareMode.OR, 1, [0, 1], 1),
        # (ECompareMode.OR, 1, [1, 0], 1),
        # (ECompareMode.OR, 1, [1, 1], 1),
        # #
        # (ECompareMode.OR, 0.4, [0, 0.5], 0),
        # (ECompareMode.OR, 0.51, [0, 0.5], 1),
        # (ECompareMode.OR, 1, [0, 1], 1),
        # (ECompareMode.OR, 0.99, [0, 1], 1),
        # (ECompareMode.OR, 1, [1, 1], 1),
        # (ECompareMode.OR, 1, [-1, 0], 0),
        # (ECompareMode.OR, 1, [-1, -1], 0),
        # (ECompareMode.OR, 1, [-1, 1], 1),
        # # ===== XOR =====
        # (ECompareMode.XOR, 0, [0, 0], 0),
        # (ECompareMode.XOR, 0, [0, 1], 0),
        # (ECompareMode.XOR, 0, [1, 0], 1),
        # (ECompareMode.XOR, 0, [1, 1], 1),
        # (ECompareMode.XOR, 1, [0, 0], 0),
        # (ECompareMode.XOR, 1, [1, 0], 1),
        # (ECompareMode.XOR, 1, [0, 1], 1),
        # (ECompareMode.XOR, 1, [1, 1], 0),
        # #
        # (ECompareMode.XOR, 0.5, [0, 0], 0),
        # (ECompareMode.XOR, 0.5, [0, 1], 1),
        # (ECompareMode.XOR, 0.5, [1, 0], 1),
        # (ECompareMode.XOR, 0.5, [1, 1], 0),
        # #
        # (ECompareMode.XOR, 0.49, [0, 0.5], 0),
        # (ECompareMode.XOR, 0.49, [1, 0.5], 1),
        # (ECompareMode.XOR, 0.51, [0, 0.5], 1),
        # (ECompareMode.XOR, 0.51, [1, 0.5], 0),
        # #
        # (ECompareMode.XOR, 0, [0, -1], 0),
        # (ECompareMode.XOR, 0, [-1, 0], 0),
        # (ECompareMode.XOR, 0, [-1, 1], 0),
        # (ECompareMode.XOR, 1, [0, -1], 0),
        # (ECompareMode.XOR, 1, [1, -1], 1),
        # (ECompareMode.XOR, 1, [-1, 1], 1),
    ],
)
def test_collapse_prompts(mode: ECompareMode, threshold: float, layer_values: list[float], expected_value: float):
    """
    Test the _collapse_prompts function with different modes and strengths.
    """
    # Prepare input tensor by broadcasting the layer values along the channel dimension into a tensor shaped (1, layer-count, 4, 4)
    expected = torch.full((1, 1, 4, 4), expected_value, dtype=torch.float32)
    layer_count = len(layer_values)
    # Create a tensor of shape (1, layer_count, 1, 1)
    base = torch.tensor(layer_values, dtype=torch.float32).reshape((1, layer_count, 1, 1))
    # Expand to (1, layer_count, 4, 4)
    tensor = base.expand(1, layer_count, 4, 4)
    assert tensor.shape == (1, layer_count, 4, 4), f"Expected shape (1, {layer_count}, 4, 4), got {tensor.shape}"

    # Call the function under test
    result = _collapse_prompts(tensor, threshold, mode)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected_value}, got {result.tolist()[0][0][0][0]}"