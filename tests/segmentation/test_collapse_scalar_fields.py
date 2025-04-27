from typing import Final

import pytest
import torch

from siscos_nodes.src.siscos_nodes.segmentation.common import (
    EMixingMode,
    collapse_scalar_fields,
)

# region: constants
# 1x1x2x2 tensors
T22_P_NORM: Final = torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]]]])
T22_P_NORM_2: Final = torch.tensor([[[[0.0, 0.0], [0.3333, 1.0]]]])
T22_PN_NORM: Final = torch.tensor([[[[-1.0, -0.3333], [0.3333, 1.0]]]])
T22_INC_0: Final = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
T22_INC_1: Final = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
T22_INC_5: Final = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
T22_0: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.0)
T22_05: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.5)
T22_02: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.2)
T22_025: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.25)
T22_075: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.75)
T22_08: Final = torch.fill(torch.empty(1, 1, 2, 2), 0.8)
T22_1: Final = torch.fill(torch.empty(1, 1, 2, 2), 1.0)
T22_2: Final = torch.fill(torch.empty(1, 1, 2, 2), 2.0)
T22_0_1: Final = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
T22_1_0: Final = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
T22_1_1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
T22_0_05: Final = torch.tensor([[[[0.0, 0.0], [0.5, 0.5]]]])
T22_1_05: Final = torch.tensor([[[[1.0, 1.0], [0.5, 0.5]]]])

# 1x2x2x2 tensors
T222_P_NORM: Final = torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]]])
T222_0_0: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
T222_0_1: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]])
T222_1_0: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
T222_1_1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]])
T222_0_05: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]]])
T222_1_05: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]])
T222_2_2: Final = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]])

T222_0_N1: Final = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]]]])
T222_N1_0: Final = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
T222_N1_N1: Final = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]])
T222_1_N1: Final = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]])
T222_N1_1: Final = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]])

# 1x4x2x2 tensors
T422_P_NORM: Final = torch.tensor([[[[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]], [[0.0, 0.3333], [0.6667, 1.0]]]])
T422_P_NORM_2: Final = torch.tensor([[[[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]], [[0.0, 0.0], [0.3333, 1.0]]]])
T422_PN_NORM: Final = torch.tensor([[[[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]], [[-1.0, -0.3333], [0.3333, 1.0]]]])
T422_0_0_0_0: Final = torch.fill(torch.empty(1, 4, 2, 2), 0.0)
T422_1_1_1_1: Final = torch.fill(torch.empty(1, 4, 2, 2), 1.0)
T422_2_2_2_2: Final = torch.fill(torch.empty(1, 4, 2, 2), 2.0)
T422_INC_0: Final = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]]])
T422_INC_1: Final = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]]])
# endregion: constants



@pytest.mark.parametrize(
    "mode, threshold, input, expected",
    [
        # region ===== ADD =====
        # simple test cases
        pytest.param(EMixingMode.ADD, 0, T222_0_0, T22_0, id="add <0, 0>, thresh=0"),
        pytest.param(EMixingMode.ADD, 0, T222_0_1, T22_1, id="add <0, 1>, thresh=0"),
        pytest.param(EMixingMode.ADD, 0, T222_1_0, T22_1, id="add <1, 0>, thresh=0"),
        pytest.param(EMixingMode.ADD, 0, T222_1_1, T22_1, id="add <1, 1>, thresh=0"),
        # adding with threshold = 1
        pytest.param(EMixingMode.ADD, 1, T222_0_0, T22_0, id="add <0, 0>, thresh=1"),
        pytest.param(EMixingMode.ADD, 1, T222_1_0, T22_0, id="add <1, 0>, thresh=1"),
        pytest.param(EMixingMode.ADD, 1, T222_0_1, T22_0, id="add <0, 1>, thresh=1"),
        pytest.param(EMixingMode.ADD, 1, T222_1_1, T22_0, id="add <1, 1>, thresh=1"),
        # adding with threshold = 0.5
        pytest.param(EMixingMode.ADD, 0.5, T222_0_0, T22_0, id="add <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, T222_0_1, T22_1, id="add <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, T222_1_0, T22_1, id="add <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.ADD, 0.5, T222_1_1, T22_1, id="add <1, 1>, thresh=0.5"),
        # complex test cases
        pytest.param(EMixingMode.ADD, 0, T222_0_0, T22_0, id="add <0, 0>, thresh=0 (complex)"),
        # endregion
        # region ===== SUBTRACT =====
        pytest.param(EMixingMode.SUBTRACT, 0, T222_0_0, T22_0, id="sub <0, 0>, thresh=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, T222_0_1, T22_0, id="sub <0, 1>, thresh=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, T222_1_0, T22_1, id="sub <1, 0>, thresh=0"),
        pytest.param(EMixingMode.SUBTRACT, 0, T222_1_1, T22_0, id="sub <1, 1>, thresh=0"),
        pytest.param(EMixingMode.SUBTRACT, 1, T222_0_0, T22_0, id="sub <0, 0>, thresh=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, T222_1_0, T22_0, id="sub <1, 0>, thresh=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, T222_0_1, T22_0, id="sub <0, 1>, thresh=1"),
        pytest.param(EMixingMode.SUBTRACT, 1, T222_1_1, T22_0, id="sub <1, 1>, thresh=1"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, T222_0_0, T22_0, id="sub <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, T222_0_1, T22_0, id="sub <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, T222_1_0, T22_1, id="sub <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.SUBTRACT, 0.5, T222_1_1, T22_0, id="sub <1, 1>, thresh=0.5"),
        # endregion
        # region ===== MULTIPLY =====
        pytest.param(EMixingMode.MULTIPLY, 0, T222_0_0, T22_0, id="mul <0, 0>, thresh=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, T222_0_1, T22_0, id="mul <0, 1>, thresh=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, T222_1_0, T22_0, id="mul <1, 0>, thresh=0"),
        pytest.param(EMixingMode.MULTIPLY, 0, T222_1_1, T22_1, id="mul <1, 1>, thresh=0"),
        pytest.param(EMixingMode.MULTIPLY, 1, T222_0_0, T22_0, id="mul <0, 0>, thresh=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, T222_1_0, T22_0, id="mul <1, 0>, thresh=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, T222_0_1, T22_0, id="mul <0, 1>, thresh=1"),
        pytest.param(EMixingMode.MULTIPLY, 1, T222_1_1, T22_0, id="mul <1, 1>, thresh=1"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, T222_0_0, T22_0, id="mul <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, T222_0_1, T22_0, id="mul <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, T222_1_0, T22_0, id="mul <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.MULTIPLY, 0.5, T222_1_1, T22_1, id="mul <1, 1>, thresh=0.5"),
        # endregion
        # region ===== SUPPRESS =====
        pytest.param(EMixingMode.SUPPRESS, 0, T222_0_0, T22_0, id="sup <0, 0>, thresh=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, T222_0_1, T22_0, id="sup <0, 1>, thresh=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, T222_1_0, T22_1, id="sup <1, 0>, thresh=0"),
        pytest.param(EMixingMode.SUPPRESS, 0, T222_1_1, T22_0, id="sup <1, 1>, thresh=0"),
        pytest.param(EMixingMode.SUPPRESS, 1, T222_0_0, T22_0, id="sup <0, 0>, thresh=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, T222_1_0, T22_0, id="sup <1, 0>, thresh=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, T222_0_1, T22_0, id="sup <0, 1>, thresh=1"),
        pytest.param(EMixingMode.SUPPRESS, 1, T222_1_1, T22_0, id="sup <1, 1>, thresh=1"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, T222_0_0, T22_0, id="sup <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, T222_0_1, T22_0, id="sup <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, T222_1_0, T22_1, id="sup <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.5, T222_1_1, T22_0, id="sup <1, 1>, thresh=0.5"),
        pytest.param(EMixingMode.SUPPRESS, 0.2, T222_1_0, T22_1, id="sup <1, 1>, thresh=0.2"),
        pytest.param(EMixingMode.SUPPRESS, 0.2, T222_1_1, T22_0, id="sup <1, 1>, thresh=0.2"),
        pytest.param(EMixingMode.SUPPRESS, 0.8, T222_1_0, T22_1, id="sup <1, 1>, thresh=0.8"),
        pytest.param(EMixingMode.SUPPRESS, 0.8, T222_1_1, T22_0, id="sup <1, 1>, thresh=0.8"),
        # 
        pytest.param(EMixingMode.SUPPRESS, 0, T222_0_N1, T22_0, id="sup <0, -1>, thresh=0"),
        # endregion
        # region ===== AVERAGE =====
        pytest.param(EMixingMode.AVERAGE, 0, T222_0_0, T22_0, id="avg <0, 0>, thresh=0"),
        pytest.param(EMixingMode.AVERAGE, 0, T222_0_1, T22_1, id="avg <0, 1>, thresh=0"),
        pytest.param(EMixingMode.AVERAGE, 0, T222_1_0, T22_1, id="avg <1, 0>, thresh=0"),
        pytest.param(EMixingMode.AVERAGE, 0, T222_1_1, T22_1, id="avg <1, 1>, thresh=0"),
        pytest.param(EMixingMode.AVERAGE, 1, T222_0_0, T22_0, id="avg <0, 0>, thresh=1"),
        pytest.param(EMixingMode.AVERAGE, 1, T222_1_0, T22_0, id="avg <1, 0>, thresh=1"),
        pytest.param(EMixingMode.AVERAGE, 1, T222_0_1, T22_0, id="avg <0, 1>, thresh=1"),
        pytest.param(EMixingMode.AVERAGE, 1, T222_1_1, T22_0, id="avg <1, 1>, thresh=1"),
        pytest.param(EMixingMode.AVERAGE, 0.5, T222_0_0, T22_0, id="avg <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, T222_0_1, T22_1, id="avg <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, T222_1_0, T22_1, id="avg <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.AVERAGE, 0.5, T222_1_1, T22_1, id="avg <1, 1>, thresh=0.5"),
        # endregion
        # region ===== MIN =====
        pytest.param(EMixingMode.MIN, 0, T222_0_0, T22_0, id="min <0, 0>, thresh=0"),
        pytest.param(EMixingMode.MIN, 0, T222_0_1, T22_0, id="min <0, 1>, thresh=0"),
        pytest.param(EMixingMode.MIN, 0, T222_1_0, T22_0, id="min <1, 0>, thresh=0"),
        pytest.param(EMixingMode.MIN, 0, T222_1_05, T22_0, id="min <1, 0.5>, thresh=0"),
        pytest.param(EMixingMode.MIN, 1, T222_0_0, T22_0, id="min <0, 0>, thresh=1"),
        pytest.param(EMixingMode.MIN, 1, T222_1_0, T22_0, id="min <1, 0>, thresh=1"),
        pytest.param(EMixingMode.MIN, 1, T222_0_05, T22_0, id="min <0, 0.5>, thresh=1"),
        pytest.param(EMixingMode.MIN, 1, T222_1_05, T22_0, id="min <1, 0.5>, thresh=1"),
        # endregion
        # region ===== MAX =====
        pytest.param(EMixingMode.MAX, 0, T222_0_0, T22_0, id="max <0, 0>, thresh=0"),
        pytest.param(EMixingMode.MAX, 0, T222_0_05, T22_0, id="max <0, 0.5>, thresh=0"),
        pytest.param(EMixingMode.MAX, 0, T222_1_0, T22_1, id="max <1, 0>, thresh=0"),
        pytest.param(EMixingMode.MAX, 0, T222_1_05, T22_1, id="max <1, 0.5>, thresh=0"),
        pytest.param(EMixingMode.MAX, 1, T222_0_0, T22_0, id="max <0, 0>, thresh=1"),
        pytest.param(EMixingMode.MAX, 1, T222_1_0, T22_0, id="max <1, 0>, thresh=1"),
        pytest.param(EMixingMode.MAX, 1, T222_0_05, T22_0, id="max <0, 0.5>, thresh=1"),
        pytest.param(EMixingMode.MAX, 1, T222_1_05, T22_0, id="max <1, 0.5>, thresh=1"),
        # endregion
        # region ===== AND =====
        pytest.param(EMixingMode.AND, 0, T222_0_0, T22_0, id="and <0, 0>, thresh=0"),
        pytest.param(EMixingMode.AND, 0, T222_0_1, T22_0, id="and <0, 1>, thresh=0"),
        pytest.param(EMixingMode.AND, 0, T222_1_0, T22_0, id="and <1, 0>, thresh=0"),
        pytest.param(EMixingMode.AND, 0, T222_1_1, T22_0, id="and <1, 1>, thresh=0"),
        pytest.param(EMixingMode.AND, 1, T222_0_0, T22_0, id="and <0, 0>, thresh=1"),
        pytest.param(EMixingMode.AND, 1, T222_1_0, T22_0, id="and <1, 0>, thresh=1"),
        pytest.param(EMixingMode.AND, 1, T222_0_1, T22_0, id="and <0, 1>, thresh=1"),
        pytest.param(EMixingMode.AND, 1, T222_1_1, T22_0, id="and <1, 1>, thresh=1"),
        pytest.param(EMixingMode.AND, 0.5, T222_0_0, T22_0, id="and <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.AND, 0.5, T222_0_1, T22_0, id="and <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.AND, 0.5, T222_1_0, T22_0, id="and <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.AND, 0.5, T222_1_1, T22_1, id="and <1, 1>, thresh=0.5"),
        #
        pytest.param(EMixingMode.AND, 0.49, T222_0_05, T22_0, id="and <0, 0.5>, thresh=0.49"),
        pytest.param(EMixingMode.AND, 0.49, T222_1_05, T22_0, id="and <1, 0.5>, thresh=0.49"),
        pytest.param(EMixingMode.AND, 0.51, T222_0_05, T22_0, id="and <0, 0.5>, thresh=0.51"),
        pytest.param(EMixingMode.AND, 0.51, T222_1_05, T22_1, id="and <1, 0.5>, thresh=0.51"),
        #
        pytest.param(EMixingMode.AND, 0, T222_0_N1, T22_0, id="and <0, -1>, thresh=0"),
        pytest.param(EMixingMode.AND, 0, T222_N1_0, T22_0, id="and <-1, 0>, thresh=0"),
        pytest.param(EMixingMode.AND, 0, T222_N1_1, T22_0, id="and <-1, 1>, thresh=0"),
        pytest.param(EMixingMode.AND, 1, T222_0_N1, T22_0, id="and <0, -1>, thresh=1"),
        pytest.param(EMixingMode.AND, 1, T222_1_N1, T22_0, id="and <1, -1>, thresh=1"),
        pytest.param(EMixingMode.AND, 1, T222_N1_1, T22_0, id="and <-1, 1>, thresh=1"),
        # endregion
        # region ===== OR =====
        pytest.param(EMixingMode.OR, 0, T222_0_0, T22_0, id="or <0, 0>, thresh=0"),
        pytest.param(EMixingMode.OR, 0, T222_0_1, T22_0, id="or <0, 1>, thresh=0"),
        pytest.param(EMixingMode.OR, 0, T222_1_0, T22_1, id="or <1, 0>, thresh=0"),
        pytest.param(EMixingMode.OR, 0, T222_1_1, T22_1, id="or <1, 1>, thresh=0"),
        pytest.param(EMixingMode.OR, 1, T222_0_0, T22_0, id="or <0, 0>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_0_1, T22_0, id="or <0, 1>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_1_0, T22_0, id="or <1, 0>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_1_1, T22_0, id="or <1, 1>, thresh=1"),
        #
        pytest.param(EMixingMode.OR, 0.4, T222_0_05, T22_0, id="or <0, 0.5>, thresh=0.4"),
        pytest.param(EMixingMode.OR, 0.51, T222_0_05, T22_1, id="or <0, 0.5>, thresh=0.51"),
        pytest.param(EMixingMode.OR, 1, T222_0_1, T22_1, id="or <0, 1>, thresh=1"),
        pytest.param(EMixingMode.OR, 0.99, T222_0_1, T22_1, id="or <0, 1>, thresh=0.99"),
        pytest.param(EMixingMode.OR, 1, T222_1_1, T22_1, id="or <1, 1>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_N1_0, T22_0, id="or <-1, 0>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_N1_N1, T22_0, id="or <-1, -1>, thresh=1"),
        pytest.param(EMixingMode.OR, 1, T222_N1_1, T22_1, id="or <-1, 1>, thresh=1"),
        # endregion
        # region ===== XOR =====
        pytest.param(EMixingMode.XOR, 0, T222_0_0, T22_0, id="xor <0, 0>, thresh=0"),
        pytest.param(EMixingMode.XOR, 0, T222_0_1, T22_0, id="xor <0, 1>, thresh=0"),
        pytest.param(EMixingMode.XOR, 0, T222_1_0, T22_1, id="xor <1, 0>, thresh=0"),
        pytest.param(EMixingMode.XOR, 0, T222_1_1, T22_1, id="xor <1, 1>, thresh=0"),
        pytest.param(EMixingMode.XOR, 1, T222_0_0, T22_0, id="xor <0, 0>, thresh=1"),
        pytest.param(EMixingMode.XOR, 1, T222_1_0, T22_0, id="xor <1, 0>, thresh=1"),
        pytest.param(EMixingMode.XOR, 1, T222_0_1, T22_0, id="xor <0, 1>, thresh=1"),
        pytest.param(EMixingMode.XOR, 1, T222_1_1, T22_0, id="xor <1, 1>, thresh=1"),
        #
        pytest.param(EMixingMode.XOR, 0.5, T222_0_0, T22_0, id="xor <0, 0>, thresh=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, T222_0_1, T22_1, id="xor <0, 1>, thresh=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, T222_1_0, T22_1, id="xor <1, 0>, thresh=0.5"),
        pytest.param(EMixingMode.XOR, 0.5, T222_1_1, T22_0, id="xor <1, 1>, thresh=0.5"),
        #
        pytest.param(EMixingMode.XOR, 0.49, T222_0_05, T22_0, id="xor <0, 0.5>, thresh=0.49"),
        pytest.param(EMixingMode.XOR, 0.49, T222_1_05, T22_1, id="xor <1, 0.5>, thresh=0.49"),
        pytest.param(EMixingMode.XOR, 0.51, T222_0_05, T22_1, id="xor <0, 0.5>, thresh=0.51"),
        pytest.param(EMixingMode.XOR, 0.51, T222_1_05, T22_0, id="xor <1, 0.5>, thresh=0.51"),
        #
        pytest.param(EMixingMode.XOR, 0, T222_0_N1, T22_0, id="xor <0, -1>, thresh=0"),
        pytest.param(EMixingMode.XOR, 0, T222_N1_0, T22_0, id="xor <-1, 0>, thresh=0"),
        pytest.param(EMixingMode.XOR, 0, T222_N1_1, T22_0, id="xor <-1, 1>, thresh=0"),
        pytest.param(EMixingMode.XOR, 1, T222_0_N1, T22_0, id="xor <0, -1>, thresh=1"),
        pytest.param(EMixingMode.XOR, 1, T222_1_N1, T22_1, id="xor <1, -1>, thresh=1"),
        pytest.param(EMixingMode.XOR, 1, T222_N1_1, T22_1, id="xor <-1, 1>, thresh=1"),
        # endregion
    ],
)
def test_collapse_scalar_fields(mode: EMixingMode, threshold: float, input: torch.Tensor, expected: torch.Tensor):
    """
    Test the collapse_gradient_masks function with different modes and strengths.
    """
    # Call the function under test
    result = collapse_scalar_fields(input, threshold, mode)

    # Check the result
    assert result.shape == expected.shape, f"Expected shape {expected.shape}, got {result.shape}"
    # Check if the result is close (within epsilon) to the expected value
    assert torch.allclose(result, expected, atol=1e-6), f"""Received: {result}
    Expected: {expected}"""