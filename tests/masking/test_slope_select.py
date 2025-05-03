from typing import Final

import pytest
import torch

from siscos_nodes.src.siscos_nodes.masking.nodes import (
    mask_slope_select,
)

# region: constants
# region mock tensors
T43_1: Final = torch.tensor([
    [ 0.0, 0.2, -0.1,  0.5],
    [ 0.3, 0.8,  0.1, -0.2],
    [-0.1, 0.0,  0.4,  0.6],
])

T43_2: Final = torch.tensor([
    [0.0, 0.2, 0.0, 0.4],
    [0.3, 0.8, 0.1, 0.0],
    [0.0, 0.0, 0.4, 0.6],
])

T43_3: Final = torch.tensor([
    [0.3, 0.2, 0.0, 0.4],
    [0.3, 0.8, 0.0, 0.5],
    [0.3, 0.0, 0.4, 0.6],
])

T55_1: Final = torch.tensor([
    [0.3, 0.3, 0.1, 0.3, 0.3],
    [0.3, 0.0, 0.1, 0.0, 0.3],
    [0.1, 0.1, 0.8, 0.1, 0.1],
    [0.3, 0.0, 0.1, 0.0, 0.3],
    [0.3, 0.3, 0.1, 0.3, 0.3],
])

T77_1: Final = torch.tensor([
    [0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1],
    [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
    [0.0, 0.0, 0.2, 0.4, 0.2, 0.0, 0.0],
    [0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.0],
    [0.0, 0.0, 0.2, 0.4, 0.2, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
    [0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1],
])
# endregion
# endregion: constants


@pytest.mark.parametrize(
    "threshold, input, expected",
    [
        #region: 4x3 tensors
        pytest.param(
            1.0,
            T43_1,
            torch.zeros_like(T43_1).bool(),
            id="Test 1: Threshold 1.0",
        ),
        pytest.param(
            0.5,
            T43_1,
            torch.ones_like(T43_1).bool(),
            id="Test 2: Threshold 0.5",
        ),
        pytest.param(
            0.0,
            T43_1,
            torch.ones_like(T43_1).bool(),
            id="Test 3: Threshold 0.0",
        ),
        pytest.param(
            0.8,
            T43_2,
            torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ]).bool(),
            id="Test 4: Threshold 0.8",
        ),
        pytest.param(
            0.5,
            T43_2,
            torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]).bool(),
            id="Test 5: Threshold 0.5",
        ),
        pytest.param(
            0.0,
            T43_2,
            torch.tensor([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]).bool(),
            id="Test 6: Threshold 0.0",
        ),
        pytest.param(
            0.8,
            T43_3,
            torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ]).bool(),
            id="Test 7: Threshold 0.8",
        ),
        pytest.param(
            0.5,
            T43_3,
            torch.tensor([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]).bool(),
            id="Test 8: Threshold 0.5",
        ),
        #endregion

        #region: 5x5 tensors
        pytest.param(
            0.8,
            T55_1,
            torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]).bool(),
            id="Test 9: Threshold 0.8",
        ),
        #endregion

        #region: 7x7 tensors
        pytest.param(
            0.8,
            T77_1,
            torch.tensor([
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
            ]).bool(),
            id="Test 10: Threshold 0.8",
        ),
        #endregion
    ],
)
def test_slope_select(threshold: float, input: torch.Tensor, expected: torch.Tensor):
    """
    Test the slope_select nodes implementation logic.
    """
    # Call the function under test
    result: torch.Tensor = mask_slope_select.MaskSlopeSelectInvocation.slope_floodfill(input, threshold)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the boolean tensor results match the expected tensor
    assert torch.equal(result, expected), f"""
    Result: {result}
    Expected: {expected}"""