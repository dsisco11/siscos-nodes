from enum import Enum
from typing import Any, Literal


class EMaskingMode(str, Enum):
    """Enum for Masking Modes, which determine the form of a given image mask."""

    BOOLEAN = "Boolean" # eg: boolean tensor, shape: [1, H, W], dtype: bool, image mode = '1'
    GRADIENT = "Gradient" # eg: scalar(0.0 - 1.0) tensor, shape: [1, H, W], dtype: float8/16/32, image mode='I;16'
    IMAGE_ALPHA = "image_alpha" # eg: image mode='RGBA', dtype: uint8, alpha channel is the mask
    IMAGE_COMPOUND = "image_compound"# eg: each channel is a different mask
    IMAGE_LUMINANCE = "image_luminance" # eg: image mode='L', dtype: uint8

LMaskingMode = Literal[EMaskingMode.BOOLEAN, EMaskingMode.GRADIENT, EMaskingMode.IMAGE_ALPHA, EMaskingMode.IMAGE_COMPOUND, EMaskingMode.IMAGE_LUMINANCE]


class EMaskShape(str, Enum):
    SOLID = "solid",
    GRADIENT_RADIAL = "gradient_radial",
    GRADIENT_VERTICAL = "gradient_vertical",
    GRADIENT_HORIZONTAL = "gradient_horizontal",

LMaskShape = Literal[EMaskShape.SOLID, EMaskShape.GRADIENT_RADIAL, EMaskShape.GRADIENT_VERTICAL, EMaskShape.GRADIENT_HORIZONTAL]

class EMathOperators(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AVERAGE = "average"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    XOR = "xor"
    OR = "or"
    AND = "and"

LMathOperators = Literal[EMathOperators.ADD, EMathOperators.SUBTRACT, EMathOperators.MULTIPLY, EMathOperators.DIVIDE, EMathOperators.AVERAGE, EMathOperators.MEDIAN, EMathOperators.MAX, EMathOperators.MIN, EMathOperators.XOR, EMathOperators.OR, EMathOperators.AND]