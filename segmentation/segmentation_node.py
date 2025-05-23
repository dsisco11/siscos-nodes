from enum import Enum
from typing import Literal

import torch
from torchvision.transforms.functional import to_pil_image as tensor_to_image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext, MetadataField
from invokeai.invocation_api import WithBoard

from ..primitives import AdvancedMaskOutput, EMaskMode, MaskField
from ..tensor_common import (
    apply_feathering_ellipse,
    gaussian_blur,
    normalize_tensor,
    upscale_tensor,
)
from .clipseg import CLIPSegSegmentationModel
from .groupvit import GroupVitSegmentationModel
from .segmentation_model import SegmentationModel


class ESegmentationModel(str, Enum):
    """Defines the available image-segmentation models."""
    CLIPSEG = "clipseg"
    GROUP_VIT = "groupvit"

class ESegmentationBlendMode(str, Enum):
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
    NOT = "not"

# TODO:(sisco): Figure out a way to use the enums in the type hints for the input fields instead of these gross literals...
SegmentationBlendMode = Literal["suppress", "subtract", "add", "multiply", "average", "max", "min", "xor", "or", "and", "not"]
SegmentationModelType = Literal["clipseg", "groupvit"]
SEGMENTATION_MODEL_TYPES: dict[ESegmentationModel, type[SegmentationModel]] = {
    ESegmentationModel.CLIPSEG: CLIPSegSegmentationModel,
    ESegmentationModel.GROUP_VIT: GroupVitSegmentationModel,
}

@torch.compile(dynamic=True)
def _blend_logits(blend_mode: str, lhs: torch.Tensor, rhs: torch.Tensor, blend_factor: float) -> torch.Tensor:
    """Blend two masks together using the specified blend mode."""
    match ESegmentationBlendMode(blend_mode):
        case ESegmentationBlendMode.SUPPRESS:
            # suppress the positive mask according to the inverse of the negative mask
            return torch.mul(lhs, torch.sub(1.0, rhs).mul(blend_factor))
        case ESegmentationBlendMode.SUBTRACT:
            return torch.sub(lhs, torch.mul(rhs, blend_factor)).clamp(min=0.0, max=1.0)
        case ESegmentationBlendMode.ADD:
            return torch.add(lhs, torch.mul(rhs, blend_factor)).clamp(min=0.0, max=1.0)
        case ESegmentationBlendMode.MULTIPLY:
            return torch.mul(lhs, torch.mul(rhs, blend_factor))
        case ESegmentationBlendMode.AVERAGE:
            # average the two masks together
            return torch.mean(torch.stack([lhs, rhs]), dim=0).softmax(dim=1).clamp(min=0.0, max=1.0)
        case ESegmentationBlendMode.MAX:
            # take the maximum of the two masks
            return torch.max(lhs, torch.mul(rhs, blend_factor))
        case ESegmentationBlendMode.MIN:
            # take the minimum of the two masks
            return torch.min(lhs, torch.mul(rhs, blend_factor))
        case ESegmentationBlendMode.XOR:
            # take the exclusive or of the two masks
            return torch.logical_xor(lhs, torch.mul(rhs, blend_factor)).float()
        case ESegmentationBlendMode.OR:
            # take the logical or of the two masks
            return torch.logical_or(lhs, torch.mul(rhs, blend_factor)).float()
        case ESegmentationBlendMode.AND:
            # take the logical and of the two masks
            return torch.logical_and(lhs, torch.mul(rhs, blend_factor)).float()
        case ESegmentationBlendMode.NOT:
            # take the logical not of the first mask
            return torch.logical_not(lhs).float()
        case _:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

@torch.compile(dynamic=True)
def _postprocess_logits(tensor: torch.Tensor, min_threshold: float) -> torch.Tensor:
    return normalize_tensor(tensor.mean(dim=1, keepdim=True).subtract(min_threshold).clamp(min=0.0))

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
            "clipseg": "CLIPSeg",
            "groupvit": "GroupViT",
        }
    )
    blend_mode: SegmentationBlendMode = InputField(
        default=ESegmentationBlendMode.SUPPRESS, description="How to blend the positive and negative prompts"
    )
    blend_factor: float = InputField(
        default=1.0, description="The blend factor to use for blending the positive and negative prompts"
    )
    smoothing: float = InputField(default=4.0, title="Smoothing", description="Smoothing radius to apply to the raw segmentation response")
    min_threshold: float = InputField(title="Minimum Threshold",
        description="The minimum threshold to use for the positive/negative response. Values below this will be clipped to 0 for both",
        default=0.0,
    )
    mask_feathering: float = InputField(title="Mask Feathering",
        default=2, description="Feathering radius for the mask. 0 means no feathering"
    )
    mask_blur: float = InputField(default=2.0, title="Mask Blur", description="Blur radius for the mask. 0 means no blur")
    prompt_positive: list[str] = InputField(title="Positive Prompt", description="The positive prompt(s) to use for segmentation.\nResults from all positive prompts are averaged together before being affected by the negatives.")
    prompt_negative: list[str] = InputField(title="Negative Prompt", description="The negative prompt(s) to use for segmentation.\nResults from all negative prompts are averaged together before affecting the positives.")
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
            mask_out = tensor_to_image(mask_tensor, mode="L")
            mask_dto:ImageDTO = context.images.save(image=mask_out, image_category=ImageCategory.MASK)
            return AdvancedMaskOutput(
                mask=MaskField(asset_id=mask_dto.image_name, mode=EMaskMode.BOOLEAN),
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

        pos_logits = _postprocess_logits(logits[:, :pos_prompt_count], self.min_threshold)   # (B, 1, H₁, W₁)
        net_logits = pos_logits

        neg_logits: torch.Tensor = None
        if (neg_prompt_count > 0):
            neg_logits = _postprocess_logits(logits[:, pos_prompt_count:], self.min_threshold)  # (B, 1, H₁, W₁)
        else:
            neg_logits = torch.zeros_like(pos_logits)

        net_logits = _blend_logits(self.blend_mode, pos_logits, neg_logits, self.blend_factor)

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
        if (self.mask_feathering > 0):
            context.util.signal_progress("Feathering mask", 0.7)
            net_logits = apply_feathering_ellipse(net_logits, self.mask_feathering)

        if (self.mask_blur > 0):
            context.util.signal_progress("Blurring mask", 0.8)
            net_logits = gaussian_blur(net_logits, sigma=self.mask_blur)

        # Squeeze the channel dimension.
        net_logits = net_logits.squeeze(0)
        _, height, width = net_logits.shape

        context.util.signal_progress("Finalizing mask", 0.9)
        image_out = tensor_to_image(net_logits, mode="L")
        _metadata = MetadataField({
            "origin": self.image.image_name,
            "segmentation_model": self.model_type,
            "positive_prompt": self.prompt_positive,
            "negative_prompt": self.prompt_negative,
            "blend_mode": self.blend_mode,
            "blend_factor": self.blend_factor,
            "smoothing": self.smoothing,
            "min_threshold": self.min_threshold,
            "mask_feathering": self.mask_feathering,
            "mask_blur": self.mask_blur,
        })
        mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK, metadata=_metadata)
        context.util.signal_progress("Finished", 1)
        return AdvancedMaskOutput(
            mask=MaskField(asset_id=mask_dto.image_name, mode=EMaskMode.IMAGE_LUMINANCE),
            image=ImageField(image_name=mask_dto.image_name),
            width=width,
            height=height,
        )
