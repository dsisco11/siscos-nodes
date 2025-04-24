from enum import Enum
from typing import Literal

import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    MetadataField,
)
from invokeai.invocation_api import WithBoard
from torchvision.transforms.functional import to_pil_image as tensor_to_pil

from ...util.primitives import AdvancedMaskOutput, EMaskingMode, MaskingField
from ...util.tensor_common import (
    gaussian_blur,
    normalize_tensor,
    upscale_tensor,
)
from ..clipseg import CLIPSegSegmentationModel
from ..groupvit import GroupVitSegmentationModel
from ..segmentation_model import SegmentationModel


class ESegmentationModel(str, Enum):
    """Defines the available image-segmentation models."""
    CLIPSEG = "clipseg"
    GROUP_VIT = "groupvit"

class ECompareMode(str, Enum):
    """Defines the blending modes for segmentation masks."""
    SUPPRESS = "suppress"
    SUBTRACT = "subtract"
    ADD = "add"
    MULTIPLY = "multiply"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    XOR = "xor"
    OR = "or"
    AND = "and"

# TODO:(sisco): Figure out a way to use the enums in the type hints for the input fields instead of these gross literals...
CompareMode = Literal["suppress", "subtract", "add", "multiply", "average", "max", "min", "xor", "or", "and"]
SegmentationModelType = Literal[ESegmentationModel.CLIPSEG, ESegmentationModel.GROUP_VIT]
SEGMENTATION_MODEL_TYPES: dict[ESegmentationModel, type[SegmentationModel]] = {
    ESegmentationModel.CLIPSEG: CLIPSegSegmentationModel,
    ESegmentationModel.GROUP_VIT: GroupVitSegmentationModel,
}

def _compare_prompts(mode: ECompareMode, lhs: torch.Tensor, rhs: torch.Tensor, rhs_factor: float) -> torch.Tensor:
    """Blend two tensors together using the specified mode.
        For non-binary modes, the factor is used to scale the rhs tensor before blending.
        For binary modes (OR/AND/XOR) the factor acts as an offset for rhs, allowing it to still be used to adjust the outcome.
    """
    assert lhs.dim() == rhs.dim() == 4, f"Expected tensors to have shape [B, C, H, W], but got {lhs.shape} and {rhs.shape}"
    match mode:
        case ECompareMode.SUPPRESS:
            # suppress the positive mask according to the inverse of the negative mask
            inv = torch.sub(1.0, rhs, alpha=rhs_factor)
            return torch.mul(lhs, inv)
        case ECompareMode.SUBTRACT:
            return torch.sub(lhs, rhs, alpha=rhs_factor)
        case ECompareMode.ADD:
            return torch.add(lhs, rhs, alpha=rhs_factor)
        case ECompareMode.MULTIPLY:
            return torch.mul(lhs, (rhs * rhs_factor))
        case ECompareMode.AVERAGE:
            # average the two masks together
            return torch.mean(torch.cat((lhs, rhs * rhs_factor), dim=1), dim=1, keepdim=True)
        case ECompareMode.MAX:
            # take the maximum of the two masks
            return torch.max(lhs, (rhs * rhs_factor))
        case ECompareMode.MIN:
            # take the minimum of the two masks
            return torch.min(lhs, (rhs * rhs_factor))
        case ECompareMode.XOR:
            return torch.logical_xor(lhs.gt(0), torch.gt(rhs, 1.0 - rhs_factor)).float()
        case ECompareMode.OR:
            return torch.logical_or(lhs.gt(0), torch.gt(rhs, 1.0 - rhs_factor)).float()
        case ECompareMode.AND:
            return torch.logical_and(lhs.gt(0), torch.gt(rhs, 1.0 - rhs_factor)).float()
        case _:
            raise ValueError(f"Unknown blend mode: {mode}")

def _collapse_prompts(tensor: torch.Tensor, threshold: float, blend_mode: ECompareMode) -> torch.Tensor:
    """Collapse the prompts into a single mask layer using the specified blend mode."""
    # For these blending modes, we apply the logic across the channel dimension.
    # This is done to combine the results of multiple prompts into a single mask.
    assert tensor.dim() == 4, f"Expected tensor to have shape [B, C, H, W], but got {tensor.shape}"
    match blend_mode:
        case ECompareMode.AVERAGE:
            tensor = tensor.mean(dim=1, keepdim=True).unsqueeze(0)
        case ECompareMode.SUPPRESS:
            tensor = (tensor.amax(dim=1, keepdim=True) * (1 - tensor.amin(dim=1, keepdim=True)))
        case ECompareMode.SUBTRACT:
            # Subtract all subsequent layers after the first from the first
            if tensor.shape[1] > 1:
                sum = torch.sum(tensor[1:], dim=1, keepdim=True)
                tensor = tensor[0:1] - sum # results in [N-1, H, W]
        case ECompareMode.ADD:
            # Add all the batches together
            tensor = tensor.sum(dim=1, keepdim=True)
        case ECompareMode.MULTIPLY:
            # Multiply all the batches together
            tensor = tensor * tensor.mean(dim=1, keepdim=True)
        case ECompareMode.MAX:
            # Take the maximum of all the batches
            tensor = tensor.amax(dim=1, keepdim=True)
        case ECompareMode.MIN:
            # Take the minimum of all the batches
            tensor = tensor.amin(dim=1, keepdim=True)
        case ECompareMode.XOR:
            # Take the exclusive or of all the batches
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_xor(tensor.amax(dim=1)).float()
        case ECompareMode.OR:
            # Take the logical or of all the batches
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_or(tensor.amax(dim=1)).float()
        case ECompareMode.AND:
            # Take the logical and of all the batches
            tensor = tensor.sub(threshold).amax(dim=1).gt(0.0).logical_and(tensor.amax(dim=1)).float()
        case _:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

    # Normalize the tensor to be between 0 and 1
    tensor = normalize_tensor(tensor)
    inv = 1.0 / (1.0 - threshold)
    result = (tensor - threshold) * inv

    assert result.shape[0] == 1 and result.ndim == 4, f"Expected tensor to have shape [1, 1, H, W], but got {result.shape}"
    return result

@invocation(
    "segmentation_mask_resolver",
    title="Segmentation Resolver",
    tags=["mask", "segmentation", "txt2mask"],
    category="mask",
    version="0.0.1",
)
class ResolveSegmentationMaskInvocation(BaseInvocation, WithBoard):
    """Uses the chosen image-segmentation model to resolve a mask from the given image using a positive & negative prompt.

    Returns a mask which matches the positive prompt(s) and does not match the negative prompt(s).
    The resulting mask indicates the relative intensity of how strongly different areas of the image match the positive prompt(s) minus the negative prompt(s).

"""

    image: ImageField = InputField(title="Image", description="The image to segment")
    model_type: SegmentationModelType = InputField(title="Resolver",
        default=ESegmentationModel.CLIPSEG, description="The model to use for segmentation",
        ui_choice_labels={
            ESegmentationModel.CLIPSEG: "CLIPSeg",
            ESegmentationModel.GROUP_VIT: "GroupViT",
        }
    )
    blend_mode: CompareMode = InputField(title="Prompt Mode",
        default=ECompareMode.AVERAGE, description="How to combine prompts within the same positive/negative group amongst themselves"
    )
    compare_mode: CompareMode = InputField(title="Comparison Mode",
        default=ECompareMode.SUPPRESS, description="How to compare the positive and negative prompts"
    )
    smoothing: float = InputField(default=4.0, title="Smoothing", description="Smoothing radius to apply to the raw segmentation response")
    # mask_feathering: float = InputField(title="Mask Feathering",
    #     default=2, description="Feathering radius for the mask. 0 means no feathering"
    # )
    # mask_blur: float = InputField(default=2.0, title="Mask Blur", description="Blur radius for the mask. 0 means no blur")
    min_threshold: float = InputField(title="Threshold",
        description="The minimum threshold to use for the positive/negative response. Values below this will be clipped to 0 for both",
        default=0.0,
    )
    negative_strength: float = InputField(title="Negative Attenuation",
        default=1.0, description="""Attenuation strength of the negative prompt when blending with the positive prompt.
        For binary modes (eg: OR, AND, XOR) 
            this is an inverse-offset subtracted from the negative results.
            Formula: blend(mode, pos, neg - (1 - attenuation))
        For all other modes, this is a scaling factor applied to the negative results.
            Formula: blend(mode, pos, neg * attenuation)
        """
    )
    prompt_positive: list[str] = InputField(title="Positive Prompt", description="The positive prompt(s) to use for segmentation.\nResults from all positive prompts are combined together before being affected by the negatives.")
    prompt_negative: list[str] = InputField(title="Negative Prompt", description="The negative prompt(s) to use for segmentation.\nResults from all negative prompts are combined together before affecting the positives.")
    # TODO:(sisco): Add support for tiling.
    use_tiling: bool = InputField(
        default=False, title="Use Tiling", description="Whether to use tiling for larger images. This will split the image into tiles and process each tile separately."
    )

    def invoke(self, context: InvocationContext) -> AdvancedMaskOutput:
        image_in = context.images.get_pil(self.image.image_name, mode="RGB")
        image_size = image_in.size
        pos_prompt_count = len(self.prompt_positive)
        neg_prompt_count = len(self.prompt_negative)
        model: SegmentationModel = SEGMENTATION_MODEL_TYPES[ESegmentationModel(self.model_type)]()

        # If we have no prompts, we can skip the model call and just return a blank mask.
        if (pos_prompt_count == 0 and neg_prompt_count == 0):
            mask_tensor = torch.ones((image_size[1], image_size[0]), dtype=torch.bool)
            mask_out = tensor_to_pil(mask_tensor, mode="L")
            mask_dto:ImageDTO = context.images.save(image=mask_out, image_category=ImageCategory.MASK)
            return AdvancedMaskOutput(
                mask=MaskingField(asset_id=mask_dto.image_name, mode=EMaskingMode.BOOLEAN),
                image=ImageField(image_name=mask_dto.image_name),
                width=image_size[0],
                height=image_size[1],
            )

        if (pos_prompt_count == 0):
            # Done to test if the model is working correctly
            # add a blank positive prompt to the list of prompts so we can check if the final mask is zeroed out as expected.
            pos_prompt_count = 1
            self.prompt_positive = [""]

        context.util.signal_progress("Running model", 0.0)
        # Combine the positive and negative prompts into a single list so we can do just a single model dispatch.
        # Strip each prompt to remove any leading/trailing whitespace and help stabilize behavior.
        _prompts = [x.strip() for x in (self.prompt_positive + self.prompt_negative)]
        logits = model.execute(context, image_in, prompts=_prompts)
        context.util.signal_progress("Processing results", 0.2)

        # === Overview ===
        # For both the positive & negative prompt lists, we take the mean of the logits across all of the prompts in said list.
        # This allows us to combine multiple isolated concepts into a single mask.
        # This is distinct from adding multiple concepts into a single prompt, which causes the model to produce a single mask which is a GRADIENT of all of the concepts.
        # Meaning that for a single prompt with multiple concepts, the first concept will be the most prominent and the last concept will be the least prominent.
        # So in order to provide more control over how multiple concepts interact, we provide this ability to "average" multiple concepts together.
        #
        # === Positive/Negative ===
        # In simple terms, the positive prompt list is what we want to see in the mask, and the negative prompt list is what we don't want to see in the mask.
        # The manner in which negative impacts positive is determined by the blend mode.
        #   "subtract" means that the negative mask is subtracted from the positive mask.
        #   "suppress" means that the positive mask is multiplied by the inverse of the negative mask.
        #       Which means the positive will be suppressed by the negative, but not completely removed.
        #   The other blend modes should be obvious from their names.
        #
        # === Normalization ===
        # The logits are normalized to be between 0 and 1, both before and after the blend.
        # This is done to ensure that the values are representative of the relative intensity of each mask.

        p_logits = logits[0:pos_prompt_count]
        n_logits = logits[pos_prompt_count:]

        pos_logits = _collapse_prompts(p_logits, self.min_threshold, ECompareMode(self.blend_mode))   # (B, 1, H₁, W₁)
        net_logits = pos_logits

        neg_logits: torch.Tensor = None
        if (neg_prompt_count > 0):
            neg_logits = _collapse_prompts(n_logits, self.min_threshold, ECompareMode(self.blend_mode))  # (B, 1, H₁, W₁)
        else:
            neg_logits = torch.zeros_like(pos_logits)

        net_logits = _compare_prompts(ECompareMode(self.compare_mode), pos_logits, neg_logits, self.negative_strength)

        context.util.signal_progress("Normalizing results", 0.4)
        # Normalize the values to be between 0 and 1
        net_logits = normalize_tensor(net_logits)

        if (self.smoothing > 0):
            context.util.signal_progress("Smoothing results", 0.5)
            net_logits = gaussian_blur(net_logits, sigma=self.smoothing)

        # Upscale the mask tensor to the original image size
        context.util.signal_progress("Upscaling mask", 0.6)
        net_logits = upscale_tensor(net_logits, target_size=image_size)

        # TODO:(sisco): Fix feathering implementation
        # if (self.mask_feathering > 0):
        #     context.util.signal_progress("Feathering mask", 0.7)
        #     net_logits = apply_feathering_ellipse(net_logits, self.mask_feathering)

        # if (self.mask_blur > 0):
        #     context.util.signal_progress("Blurring mask", 0.8)
        #     net_logits = gaussian_blur(net_logits, sigma=self.mask_blur)

        # Squeeze the channel dimension.
        net_logits = net_logits.permute(1,0,2,3).squeeze(0)
        _, height, width = net_logits.shape

        print(f"Mask shape: {net_logits.shape}")

        context.util.signal_progress("Finalizing mask", 0.9)
        image_out = tensor_to_pil(net_logits, mode="L")
        _metadata = MetadataField({
            "origin": self.image.image_name,
            "segmentation_model": self.model_type,
            "positive_prompt": self.prompt_positive,
            "negative_prompt": self.prompt_negative,
            "blend_mode": self.compare_mode,
            "blend_factor": self.negative_strength,
            "smoothing": self.smoothing,
            "min_threshold": self.min_threshold,
            # "mask_feathering": self.mask_feathering,
            # "mask_blur": self.mask_blur,
        })
        mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK, metadata=_metadata)
        context.util.signal_progress("Finished", 1)
        return AdvancedMaskOutput(
            mask=MaskingField(asset_id=mask_dto.image_name, mode=EMaskingMode.IMAGE_LUMINANCE),
            image=ImageField(image_name=mask_dto.image_name),
            width=width,
            height=height,
        )
