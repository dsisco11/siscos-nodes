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

from siscos_nodes.src.siscos_nodes.segmentation.segmentation_model import (
    SegmentationModel,
)
from siscos_nodes.src.siscos_nodes.util.primitives import (
    AdvancedMaskOutput,
    EMaskingMode,
    MaskingField,
)
from siscos_nodes.src.siscos_nodes.util.tensor_common import (
    gaussian_blur,
    normalize_tensor,
    upscale_tensor,
)

from ..common import (
    SEGMENTATION_MODEL_TYPES,
    EMixingMode,
    ESegmentationModel,
    MixingMode,
    SegmentationModelType,
    collapse_scalar_fields,
    compare_scalar_fields,
)


@invocation(
    "segmentation_mask_resolver",
    title="Segmentation Resolver",
    tags=["mask", "segmentation", "txt2mask"],
    category="mask",
    version="0.1.0",
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
    blend_mode: MixingMode = InputField(title="Prompt Mode",
        default=EMixingMode.AVERAGE, description="How to combine prompts within the same positive/negative group amongst themselves"
    )
    compare_mode: MixingMode = InputField(title="Comparison Mode",
        default=EMixingMode.SUPPRESS, description="How to compare the positive and negative prompts"
    )
    smoothing: float = InputField(default=4.0, title="Smoothing", description="Smoothing radius to apply to the raw segmentation response")
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

    @torch.no_grad()
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
        p_logits = logits[0:pos_prompt_count].permute(1, 0, 2, 3)
        n_logits = logits[pos_prompt_count:].permute(1, 0, 2, 3)

        pos_logits = collapse_scalar_fields(p_logits, self.min_threshold, EMixingMode(self.blend_mode))   # (B, 1, H₁, W₁)
        net_logits = pos_logits

        neg_logits: torch.Tensor = None
        if (neg_prompt_count > 0):
            neg_logits = collapse_scalar_fields(n_logits, self.min_threshold, EMixingMode(self.blend_mode))  # (B, 1, H₁, W₁)
        else:
            neg_logits = torch.zeros_like(pos_logits)

        net_logits = compare_scalar_fields(EMixingMode(self.compare_mode), pos_logits, neg_logits, self.negative_strength)

        if (self.smoothing > 0):
            context.util.signal_progress("Smoothing results", 0.5)
            net_logits = gaussian_blur(net_logits, sigma=self.smoothing)

        # Upscale the mask tensor to the original image size
        context.util.signal_progress("Upscaling mask", 0.6)
        net_logits = upscale_tensor(net_logits, target_size=image_size)

        # Squeeze the channel dimension.
        net_logits = net_logits.permute(1,0,2,3).squeeze(0)
        _, height, width = net_logits.shape

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
            "min_threshold": self.min_threshold
        })
        mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK, metadata=_metadata)
        context.util.signal_progress("Finished", 1)
        return AdvancedMaskOutput(
            mask=MaskingField(asset_id=mask_dto.image_name, mode=EMaskingMode.IMAGE_LUMINANCE),
            image=ImageField(image_name=mask_dto.image_name),
            width=width,
            height=height,
        )
