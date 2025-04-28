import pytest
import torch

from siscos_nodes.src.siscos_nodes.segmentation.nodes.segmentation_node import (
    EMixingMode,
    compare_scalar_fields,
)


@pytest.mark.parametrize(
    "mode, strength, lhs, rhs, expected_value",
    [
        # region ADD
        pytest.param(EMixingMode.ADD, 0, 0, 0, 0, id="add 0+0, s=0"),
        pytest.param(EMixingMode.ADD, 0, 0, 1, 0, id="add 0+1, s=0"),
        pytest.param(EMixingMode.ADD, 0, 1, 0, 1, id="add 1+0, s=0"),
        pytest.param(EMixingMode.ADD, 0, 1, 1, 1, id="add 1+1, s=0"),
        pytest.param(EMixingMode.ADD, 1, 0, 0, 0, id="add 0+0, s=1"),
        pytest.param(EMixingMode.ADD, 1, 1, 0, 1, id="add 1+0, s=1"),
        pytest.param(EMixingMode.ADD, 1, 0, 1, 1, id="add 0+1, s=1"),
        pytest.param(EMixingMode.ADD, 1, 1, 1, 2, id="add 1+1, s=1"),
        pytest.param(EMixingMode.ADD, 0.5, 0, 0, 0, id="add 0+0, s=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, 0, 1, 0.5, id="add 0+1, s=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, 1, 0, 1, id="add 1+0, s=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, 1, 1, 1.5, id="add 1+1, s=0.5"),
        # endregion
        # region SUBTRACT
        pytest.param(EMixingMode.SUBTRACT, 0, 0, 0, 0, id="sub 0-0, s=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, 0, 1, 0, id="sub 0-1, s=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, 1, 0, 1, id="sub 1-0, s=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, 1, 1, 1, id="sub 1-1, s=0"),
        pytest.param(EMixingMode.SUBTRACT, 1, 0, 0, 0, id="sub 0-0, s=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, 1, 0, 1, id="sub 1-0, s=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, 0, 1, 0, id="sub 0-1, s=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, 1, 1, 0, id="sub 1-1, s=1"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, 0, 0, 0, id="sub 0-0, s=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, 0, 1, 0, id="sub 0-1, s=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, 1, 0, 1, id="sub 1-0, s=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, 1, 1, 0.5, id="sub 1-1, s=0.5"),
        # endregion
        # region MULTIPLY
        pytest.param(EMixingMode.MULTIPLY, 0, 0, 0, 0, id="mul 0*0, s=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, 0, 1, 0, id="mul 0*1, s=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, 1, 0, 0, id="mul 1*0, s=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, 1, 1, 0, id="mul 1*1, s=0"),
        pytest.param(EMixingMode.MULTIPLY, 1, 0, 0, 0, id="mul 0*0, s=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, 1, 0, 0, id="mul 1*0, s=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, 0, 1, 0, id="mul 0*1, s=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, 1, 1, 1, id="mul 1*1, s=1"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, 0, 0, 0, id="mul 0*0, s=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, 0, 1, 0, id="mul 0*1, s=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, 1, 0, 0, id="mul 1*0, s=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, 1, 1, 0.5, id="mul 1*1, s=0.5"),
        # endregion
        # region SUPPRESS
        pytest.param(EMixingMode.SUPPRESS, 0, 0, 0, 0, id="sup 0!0, s=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, 0, 1, 0, id="sup 0!1, s=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, 1, 0, 1, id="sup 1!0, s=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, 1, 1, 1, id="sup 1!1, s=0"),
        pytest.param(EMixingMode.SUPPRESS, 1, 0, 0, 0, id="sup 0!0, s=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, 1, 0, 1, id="sup 1!0, s=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, 0, 1, 0, id="sup 0!1, s=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, 1, 1, 0, id="sup 1!1, s=1"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, 0, 0, 0, id="sup 0!0, s=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, 0, 1, 0, id="sup 0!1, s=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, 1, 0, 1, id="sup 1!0, s=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, 1, 1, 0.5, id="sup 1!1, s=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.2, 1, 1, 0.8, id="sup 1!1, s=0.2"),
        pytest.param(EMixingMode.SUPPRESS, 0.8, 1, 1, 0.2, id="sup 1!1, s=0.8"),
        # endregion
        # region AVERAGE
        pytest.param(EMixingMode.AVERAGE, 0, 0, 0, 0, id="avg 0,0, s=0"),
        pytest.param(EMixingMode.AVERAGE, 0, 0, 1, 0, id="avg 0,1, s=0"),
        pytest.param(EMixingMode.AVERAGE, 0, 1, 0, 0.5, id="avg 1,0, s=0"),
        pytest.param(EMixingMode.AVERAGE, 0, 1, 1, 0.5, id="avg 1,1, s=0"),
        pytest.param(EMixingMode.AVERAGE, 1, 0, 0, 0, id="avg 0,0, s=1"),
        pytest.param(EMixingMode.AVERAGE, 1, 1, 0, 0.5, id="avg 1,0, s=1"),
        pytest.param(EMixingMode.AVERAGE, 1, 0, 1, 0.5, id="avg 0,1, s=1"),
        pytest.param(EMixingMode.AVERAGE, 1, 1, 1, 1, id="avg 1,1, s=1"),
        pytest.param(EMixingMode.AVERAGE, 0.5, 0, 0, 0, id="avg 0,0, s=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, 0, 1, 0.25, id="avg 0,1, s=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, 1, 0, 0.5, id="avg 1,0, s=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, 1, 1, 0.75, id="avg 1,1, s=0.5"),
        # endregion
        # region MIN
        pytest.param(EMixingMode.MIN, 0, 0, 0, 0, id="min 0,0, s=0"),
        pytest.param(EMixingMode.MIN, 0, 0, 0, 0, id="min 0,0, s=0 (dup)"),
        pytest.param(EMixingMode.MIN, 0, 1, 0, 0, id="min 1,0, s=0"),
        pytest.param(EMixingMode.MIN, 0, 1, 0.5, 0, id="min 1,0.5, s=0"),
        pytest.param(EMixingMode.MIN, 1, 0, 0, 0, id="min 0,0, s=1"),
        pytest.param(EMixingMode.MIN, 1, 1, 0, 0, id="min 1,0, s=1"),
        pytest.param(EMixingMode.MIN, 1, 0, 0.5, 0, id="min 0,0.5, s=1"),
        pytest.param(EMixingMode.MIN, 1, 1, 0.5, 0.5, id="min 1,0.5, s=1"),
        # endregion
        # region MAX
        pytest.param(EMixingMode.MAX, 0, 0, 0, 0, id="max 0,0, s=0"),
        pytest.param(EMixingMode.MAX, 0, 0, 0.5, 0, id="max 0,0.5, s=0"),
        pytest.param(EMixingMode.MAX, 0, 1, 0, 1, id="max 1,0, s=0"),
        pytest.param(EMixingMode.MAX, 0, 1, 0.5, 1, id="max 1,0.5, s=0"),
        pytest.param(EMixingMode.MAX, 1, 0, 0, 0, id="max 0,0, s=1"),
        pytest.param(EMixingMode.MAX, 1, 1, 0, 1, id="max 1,0, s=1"),
        pytest.param(EMixingMode.MAX, 1, 0, 0.5, 0.5, id="max 0,0.5, s=1"),
        pytest.param(EMixingMode.MAX, 1, 1, 0.5, 1, id="max 1,0.5, s=1"),
        # endregion
        # region AND
        pytest.param(EMixingMode.AND, 0, 0, 0, 0, id="and 0&0, s=0"),
        pytest.param(EMixingMode.AND, 0, 0, 1, 0, id="and 0&1, s=0"),
        pytest.param(EMixingMode.AND, 0, 1, 0, 0, id="and 1&0, s=0"),
        pytest.param(EMixingMode.AND, 0, 1, 1, 0, id="and 1&1, s=0"),
        pytest.param(EMixingMode.AND, 1, 0, 0, 0, id="and 0&0, s=1"),
        pytest.param(EMixingMode.AND, 1, 1, 0, 0, id="and 1&0, s=1"),
        pytest.param(EMixingMode.AND, 1, 0, 1, 0, id="and 0&1, s=1"),
        pytest.param(EMixingMode.AND, 1, 1, 1, 1, id="and 1&1, s=1"),
        pytest.param(EMixingMode.AND, 0.5, 0, 0, 0, id="and 0&0, s=0.5"),
        pytest.param(EMixingMode.AND, 0.5, 0, 1, 0, id="and 0&1, s=0.5"),
        pytest.param(EMixingMode.AND, 0.5, 1, 0, 0, id="and 1&0, s=0.5"),
        pytest.param(EMixingMode.AND, 0.5, 1, 1, 1, id="and 1&1, s=0.5"),
        pytest.param(EMixingMode.AND, 0.49, 0, 0.5, 0, id="and 0&0.5, s=0.49"),
        pytest.param(EMixingMode.AND, 0.49, 1, 0.5, 0, id="and 1&0.5, s=0.49"),
        pytest.param(EMixingMode.AND, 0.51, 0, 0.5, 0, id="and 0&0.5, s=0.51"),
        pytest.param(EMixingMode.AND, 0.51, 1, 0.5, 1, id="and 1&0.5, s=0.51"),
        pytest.param(EMixingMode.AND, 0, 0, -1, 0, id="and 0&-1, s=0"),
        pytest.param(EMixingMode.AND, 0, -1, 0, 0, id="and -1&0, s=0"),
        pytest.param(EMixingMode.AND, 0, -1, 1, 0, id="and -1&1, s=0"),
        pytest.param(EMixingMode.AND, 1, 0, -1, 0, id="and 0&-1, s=1"),
        pytest.param(EMixingMode.AND, 1, 1, -1, 0, id="and 1&-1, s=1"),
        pytest.param(EMixingMode.AND, 1, -1, 1, 0, id="and -1&1, s=1"),
        # endregion
        # region OR
        pytest.param(EMixingMode.OR, 0, 0, 0, 0, id="or 0|0, s=0"),
        pytest.param(EMixingMode.OR, 0, 0, 1, 0, id="or 0|1, s=0"),
        pytest.param(EMixingMode.OR, 0, 1, 0, 1, id="or 1|0, s=0"),
        pytest.param(EMixingMode.OR, 0, 1, 1, 1, id="or 1|1, s=0"),
        pytest.param(EMixingMode.OR, 1, 0, 0, 0, id="or 0|0, s=1"),
        pytest.param(EMixingMode.OR, 1, 0, 1, 1, id="or 0|1, s=1"),
        pytest.param(EMixingMode.OR, 1, 1, 0, 1, id="or 1|0, s=1"),
        pytest.param(EMixingMode.OR, 1, 1, 1, 1, id="or 1|1, s=1"),
        pytest.param(EMixingMode.OR, 0.4, 0, 0.5, 0, id="or 0|0.5, s=0.4"),
        pytest.param(EMixingMode.OR, 0.51, 0, 0.5, 1, id="or 0|0.5, s=0.51"),
        pytest.param(EMixingMode.OR, 1, 0, 1, 1, id="or 0|1, s=1 (dup)"),
        pytest.param(EMixingMode.OR, 0.99, 0, 1, 1, id="or 0|1, s=0.99"),
        pytest.param(EMixingMode.OR, 1, 1, 1, 1, id="or 1|1, s=1 (dup)"),
        pytest.param(EMixingMode.OR, 1, -1, 0, 0, id="or -1|0, s=1"),
        pytest.param(EMixingMode.OR, 1, -1, -1, 0, id="or -1|-1, s=1"),
        pytest.param(EMixingMode.OR, 1, -1, 1, 1, id="or -1|1, s=1"),
        # endregion
        # region XOR
        pytest.param(EMixingMode.XOR, 0, 0, 0, 0, id="xor 0^0, s=0"),
        pytest.param(EMixingMode.XOR, 0, 0, 1, 0, id="xor 0^1, s=0"),
        pytest.param(EMixingMode.XOR, 0, 1, 0, 1, id="xor 1^0, s=0"),
        pytest.param(EMixingMode.XOR, 0, 1, 1, 1, id="xor 1^1, s=0"),
        pytest.param(EMixingMode.XOR, 1, 0, 0, 0, id="xor 0^0, s=1"),
        pytest.param(EMixingMode.XOR, 1, 1, 0, 1, id="xor 1^0, s=1"),
        pytest.param(EMixingMode.XOR, 1, 0, 1, 1, id="xor 0^1, s=1"),
        pytest.param(EMixingMode.XOR, 1, 1, 1, 0, id="xor 1^1, s=1"),
        pytest.param(EMixingMode.XOR, 0.5, 0, 0, 0, id="xor 0^0, s=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, 0, 1, 1, id="xor 0^1, s=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, 1, 0, 1, id="xor 1^0, s=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, 1, 1, 0, id="xor 1^1, s=0.5"),
        pytest.param(EMixingMode.XOR, 0.49, 0, 0.5, 0, id="xor 0^0.5, s=0.49"),
        pytest.param(EMixingMode.XOR, 0.49, 1, 0.5, 1, id="xor 1^0.5, s=0.49"),
        pytest.param(EMixingMode.XOR, 0.51, 0, 0.5, 1, id="xor 0^0.5, s=0.51"),
        pytest.param(EMixingMode.XOR, 0.51, 1, 0.5, 0, id="xor 1^0.5, s=0.51"),
        pytest.param(EMixingMode.XOR, 0, 0, -1, 0, id="xor 0^-1, s=0"),
        pytest.param(EMixingMode.XOR, 0, -1, 0, 0, id="xor -1^0, s=0"),
        pytest.param(EMixingMode.XOR, 0, -1, 1, 0, id="xor -1^1, s=0"),
        pytest.param(EMixingMode.XOR, 1, 0, -1, 0, id="xor 0^-1, s=1"),
        pytest.param(EMixingMode.XOR, 1, 1, -1, 1, id="xor 1^-1, s=1"),
        pytest.param(EMixingMode.XOR, 1, -1, 1, 1, id="xor -1^1, s=1"),
        # endregion
    ],
)
def test_compare_scalar_fields(mode: EMixingMode, strength: float, lhs: float, rhs: float, expected_value: float):
    """
    Test the _compare_prompts function with different modes and strengths.
    """
    # Prepare the input tensors
    tensor_1 = torch.full((1, 1, 2, 2), lhs, dtype=torch.float32)
    tensor_2 = torch.full((1, 1, 2, 2), rhs, dtype=torch.float32)
    expected = torch.full((1, 1, 2, 2), expected_value, dtype=torch.float32)

    # Call the function under test
    result = compare_scalar_fields(mode, tensor_1, tensor_2, strength)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected_value}, got {result.tolist()[0][0][0][0]}"