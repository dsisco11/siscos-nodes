from typing import Final

import pytest
import torch

from siscos_nodes.src.siscos_nodes.util.tensor_common import (
    normalize_logits,
    normalize_tensor,
    threshold_and_normalize_tensor,
)

# region: constants
# 1x2x2 tensors
T22_INC_NORM: Final = torch.tensor([[0.0, 0.3333], [0.6667, 1.0]])
T22_INC_NORM_2: Final = torch.tensor([[0.0, 0.0], [0.3333, 1.0]])
T22_INC_NORM_NEG: Final = torch.tensor([[-1.0, -0.3333], [0.3333, 1.0]])
T22_FILL_0: Final = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
T22_FILL_1: Final = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
T22_FILL_2: Final = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
T22_INC_0: Final = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
T22_INC_1: Final = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
T22_INC_5: Final = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 4x2x2 tensors
T422_INC_NORM: Final = torch.tensor([[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]])
T422_INC_NORM_2: Final = torch.tensor([[[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]]])
T422_INC_NORM_NEG: Final = torch.tensor([[[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]]])
T422_FILL_0: Final = torch.fill(torch.empty(4, 2, 2), 0.0)
T422_FILL_1: Final = torch.fill(torch.empty(4, 2, 2), 1.0)
T422_FILL_2: Final = torch.fill(torch.empty(4, 2, 2), 2.0)
T422_INC_0: Final = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]])
T422_INC_1: Final = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]])
# endregion: constants


@pytest.mark.parametrize(
        [
            "tensor",
            "expected_tensor",
        ],
        [
            (T22_FILL_0, T22_FILL_0),
            (T22_FILL_2, T22_FILL_1),
            (T22_INC_0, T22_INC_NORM),
            (T22_INC_1, T22_INC_NORM),
            (T22_INC_5, T22_INC_NORM),
            (T422_INC_0, T422_INC_NORM),
            (T422_INC_1, T422_INC_NORM),
        ]
)
def test_normalize_tensor(tensor: torch.Tensor, expected_tensor: torch.Tensor):
    # Normalize the tensor
    normalized_tensor = normalize_tensor(tensor)

    # Check if the normalized tensor is equal to the expected tensor
    assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-4), (
        f"Expected {expected_tensor}, but got {normalized_tensor}"
    )

@pytest.mark.parametrize(
        [
            "threshold",
            "tensor",
            "expected_tensor",
        ],
        [
            (0.0, T22_FILL_0, T22_FILL_0),
            (0.0, T22_FILL_2, T22_FILL_1),
            (0.0, T22_INC_0, T22_INC_NORM),
            (0.0, T22_INC_1, T22_INC_NORM),
            (0.0, T22_INC_5, T22_INC_NORM),
            (0.0, T422_INC_0, T422_INC_NORM),
            (0.0, T422_INC_1, T422_INC_NORM),
            #
            (0.5, T22_FILL_0, T22_FILL_0),
            (0.5, T22_FILL_1, T22_FILL_1),
            (0.5, T22_FILL_2, T22_FILL_1),
            (0.5, T22_INC_0, T22_INC_NORM_2),
            (0.5, T22_INC_1, T22_INC_NORM_2),
            (0.5, T22_INC_5, T22_INC_NORM_2),
            (0.5, T422_INC_0, T422_INC_NORM_2),
            (0.5, T422_INC_1, T422_INC_NORM_2),
            #
            (1.0, T22_FILL_0, T22_FILL_0),
            (1.0, T22_FILL_1, T22_FILL_0),
            (1.0, T22_FILL_2, T22_FILL_0),
            (1.0, T22_INC_0, T22_FILL_0),
            (1.0, T22_INC_1, T22_FILL_0),
            (1.0, T22_INC_5, T22_FILL_0),
            (1.0, T422_INC_0, T422_FILL_0),
            (1.0, T422_INC_1, T422_FILL_0),
        ]
)
def test_threshold_normalize_tensor(threshold: float, tensor: torch.Tensor, expected_tensor: torch.Tensor):
    # Normalize the tensor
    normalized_tensor = threshold_and_normalize_tensor(tensor, threshold)

    # Check if the normalized tensor is equal to the expected tensor
    assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-4), (
        f"""Expected: {expected_tensor}
            Received: {normalized_tensor}"""
    )


@pytest.mark.parametrize(
        [
            "tensor",
            "expected_tensor",
        ],
        [
            (T22_INC_0.unsqueeze(0), T22_INC_NORM_NEG.unsqueeze(0)),
            (T22_INC_1.unsqueeze(0), T22_INC_NORM_NEG.unsqueeze(0)),
            (T422_INC_0, T422_INC_NORM_NEG),
            (T422_INC_1, T422_INC_NORM_NEG),
        ]
)
def test_normalize_logits(tensor: torch.Tensor, expected_tensor: torch.Tensor):
    # Normalize the tensor
    normalized_tensor = normalize_logits(tensor)

    # Check if the normalized tensor is equal to the expected tensor
    assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-4), (
        f"Expected {expected_tensor}, but got {normalized_tensor}"
    )