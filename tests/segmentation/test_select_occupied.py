from typing import Final

import pytest
import torch

from siscos_nodes.src.siscos_nodes.segmentation.nodes import (
    select_occupied,
)
from siscos_nodes.src.siscos_nodes.segmentation.nodes.select_occupied import (
    ListOfBoundingBoxes,
)

# region: constants
# region 4x3 tensors
T43_1: Final = torch.tensor([
    [ 0.0, 0.2, -0.1,  0.5],
    [ 0.3, 0.8,  0.1, -0.2],
    [-0.1, 0.0,  0.4,  0.6],
])

T43_2: Final = torch.tensor([
    [0.0, 0.2, 0.0, 0.0],
    [0.3, 0.8, 0.1, 0.0],
    [0.0, 0.0, 0.4, 0.0],
])

T43_3: Final = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [0.3, 0.8, 0.0, 0.0],
    [0.3, 0.0, 0.4, 0.0],
])

T43_4: Final = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.1, 0.0],
    [0.0, 0.0, 0.4, 0.0],
])
# endregion
# endregion: constants


@pytest.mark.parametrize(
    "input, expected",
    [
        pytest.param(
            T43_1,
            [[0, 0, 3, 2]],
            id="Test 1",
        ),
        pytest.param(
            T43_2,
            [[0, 0, 2, 2]],
            id="Test 2",
        ),
        pytest.param(
            T43_3,
            [[0, 1, 2, 2]],
            id="Test 3",
        ),
        pytest.param(
            T43_4,
            [[1, 1, 2, 2]],
            id="Test 4",
        ),
    ],
)
def test_slope_select(input: torch.Tensor, expected: ListOfBoundingBoxes):
    # Call the function under test
    result: ListOfBoundingBoxes = select_occupied.SelectOccupiedInvocation.resolve_bounding_boxes(input).tolist()

    # Check if the boolean tensor results match the expected tensor
    assert (result == expected), f"""
    Result: {result}
    Expected: {expected}"""