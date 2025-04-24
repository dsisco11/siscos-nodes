import pytest
import torch
from src.siscos_nodes.util.tensor_common import (
    normalize_logits,
    normalize_tensor,
)


@pytest.mark.parametrize(
        [
            "tensor",
            "expected_tensor",
        ],
        [
            (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0.0, 0.3333], [0.6667, 1.0]])),
            (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[0.0, 0.3333], [0.6667, 1.0]])),
            (torch.tensor([[5, 6], [7, 8]]), torch.tensor([[0.0, 0.3333], [0.6667, 1.0]])),
            (torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]]), torch.tensor([[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]])),
            (torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]]), torch.tensor([[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]])),
            (torch.tensor([[[[0, 1], [2, 3]], [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]],
                           [[[0, 1], [2, 3]], [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]]]), 
                           torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]],
                                         [[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]]])),
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
            "tensor",
            "expected_tensor",
        ],
        [
            (torch.tensor([[[0, 1], [2, 3]], [[5, 6], [7, 8]]]), torch.tensor([[[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]]])),
        ]
)
def test_normalize_logits(tensor: torch.Tensor, expected_tensor: torch.Tensor):
    # Normalize the tensor
    normalized_tensor = normalize_logits(tensor)

    # Check if the normalized tensor is equal to the expected tensor
    assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-4), (
        f"Expected {expected_tensor}, but got {normalized_tensor}"
    )