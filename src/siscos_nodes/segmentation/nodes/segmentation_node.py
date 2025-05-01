from typing import Optional

import torch
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    MetadataField,
)
from invokeai.backend.util.devices import TorchDevice
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
    resize_tensor,
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
    version="0.6.2",
)
class ResolveSegmentationMaskInvocation(BaseInvocation, WithBoard):
    """Uses the chosen image-segmentation model to resolve a mask from the given image using a positive & negative prompt.

    Returns a mask which matches the positive prompt(s) and does not match the negative prompt(s).
    The resulting mask indicates the relative intensity of how strongly different areas of the image match the positive prompt(s) minus the negative prompt(s).
"""

    image: ImageField = InputField(ui_order=0, title="Image", description="The image to segment")
    attention_mask: Optional[MaskingField] = InputField(
        ui_order=1,
        title="Attention Mask",
        description="The attention mask to use for the image. This is used to limit the area of the image that is processed.",
        default=None
    )
    model_type: SegmentationModelType = InputField(
        ui_order=2,
        title="Resolver",
        default=ESegmentationModel.CLIPSEG,
        description="The model to use for segmentation",
        ui_choice_labels={
            ESegmentationModel.CLIPSEG: "CLIPSeg",
            ESegmentationModel.GROUP_VIT: "GroupViT",
        }
    )
    # TODO:(sisco): Add support for tiling.
    use_tiling: bool = InputField(
        ui_order=3,
        title="Use Tiling", 
        default=False, description="Whether to use tiling for larger images. This will split the image into tiles and process each tile separately."
    )
    smoothing: float = InputField(
        ui_order=4,
        title="Smoothing", 
        default=6.0, description="Smoothing radius to apply to the raw segmentation response"
    )
    min_threshold: float = InputField(
        ui_order=6,
        title="Threshold",
        description="The minimum threshold to use for the positive/negative response. Values below this will be clipped to 0 for both",
        default=0.0,
    )
    negative_strength: float = InputField(
        ui_order=7,
        title="Negative Strength",
        default=1.0, description="""Attenuation strength of the negative prompt when blending with the positive prompt.\n
        For binary modes (eg: OR, AND, XOR)\n
            this is an inverse-offset subtracted from the negative results.\n
            Formula: blend(mode, positive, negative - (1 - attenuation))\n
        For all other modes, this is a scaling factor applied to the negative results.\n
            Formula: blend(mode, positive, negative * attenuation)
        """
    )
    
    p_blend_mode: MixingMode = InputField(
        ui_order=8,
        title="Positive Blending Mode", default=EMixingMode.AVERAGE, description="How to combine the positive prompts together"
    )
    prompt_positive: list[str] = InputField(
        ui_order=9,
        title="Positive Prompt", description="The positive prompt(s) to use for segmentation.\nResults from all positive prompts are combined together before being affected by the negatives."
    )
    n_blend_mode: MixingMode = InputField(
        ui_order=10,
        title="Negative Blending Mode", default=EMixingMode.AVERAGE, description="How to combine the negative prompts together"
    )
    prompt_negative: list[str] = InputField(
        ui_order=11,
        title="Negative Prompt", description="The negative prompt(s) to use for segmentation.\nResults from all negative prompts are combined together before affecting the positives."
    )
    compare_mode: MixingMode = InputField(
        ui_order=12,
        title="Comparison Mode",
        default=EMixingMode.SUPPRESS, description="How the negatives affect the positives.\nThis is the blend mode used to combine the positive and negative masks together.",
    )
    confidence_threshold: float = InputField(
        ui_order=13,
        title="Confidence Threshold", default=1.0, description=""
    )
    final_contrast: float = InputField(
        ui_order=14,
        title="Contrast", default=1.0, description="The contrast to apply to the final mask.\nThis is applied as a power function to the final grayscale mask.\nA value of 1.0 will not change the contrast, while a value of 0.0 will make the mask completely flat.\nA value of 2.0 will double the contrast, and a value of 0.5 will halve the contrast."
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> AdvancedMaskOutput:
        device: torch.device = TorchDevice.choose_torch_device()
        dtype: torch.dtype = TorchDevice.choose_torch_dtype()
        image_in = context.images.get_pil(self.image.image_name, mode="RGB")
        image_size = image_in.size
        pos_prompt_count = len(self.prompt_positive)
        neg_prompt_count = len(self.prompt_negative)
        model: SegmentationModel = SEGMENTATION_MODEL_TYPES[ESegmentationModel(self.model_type)]()
        attn_mask: torch.Tensor
        if (self.attention_mask is not None):
            attn_mask = self.attention_mask.load(context).unsqueeze(0)  # (1, 1, H, W)
        else:
            attn_mask = torch.ones((1, 1, image_size[1], image_size[0]), dtype=dtype, device=device)

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
        logits = model.execute(context, image_in, prompts=_prompts).clamp_min(0.0)
        context.util.signal_progress("Processing results", 0.2)

        # apply the attention mask to the logits BEFORE we collapse the scalar fields.
        # This is done to ensure that the collapse mode doesnt interfere with the attention mask.
        if (self.attention_mask is not None):
            attn = attn_mask.to(device=device, dtype=dtype)
            # match the attention mask to the logits size
            if (attn.shape[2:3] != logits.shape[2:3]):
                attn = resize_tensor(attn, target_size=(logits.shape[2], logits.shape[3]))
            logits = torch.multiply(logits, attn)

        p_logits = logits[0:pos_prompt_count]
        n_logits = logits[pos_prompt_count:]

        pos_logits = collapse_scalar_fields(p_logits, self.min_threshold, EMixingMode(self.p_blend_mode))   # (B, 1, H₁, W₁)
        net_logits = pos_logits

        neg_logits: torch.Tensor
        if (neg_prompt_count > 0):
            neg_logits = collapse_scalar_fields(n_logits, self.min_threshold, EMixingMode(self.n_blend_mode))  # (B, 1, H₁, W₁)
        else:
            neg_logits = torch.zeros_like(pos_logits)

        net_logits = compare_scalar_fields(EMixingMode(self.compare_mode), pos_logits, neg_logits, self.negative_strength)

        if (self.smoothing > 0):
            context.util.signal_progress("Smoothing results", 0.5)
            net_logits = gaussian_blur(net_logits, sigma=self.smoothing)

        if (self.confidence_threshold < 1.0):
            net_logits = normalize_tensor(net_logits.clamp_max(self.confidence_threshold))

        if (self.final_contrast != 1.0):
            context.util.signal_progress("Adjusting contrast", 0.7)
            net_logits = net_logits.pow(self.final_contrast)

        # Re-Normalize the logits
        net_logits = normalize_tensor(net_logits)

        # Upscale the mask tensor to the original image size
        context.util.signal_progress("Upscaling mask", 0.6)
        net_logits = resize_tensor(net_logits, target_size=image_size)
        
        # match the attention mask to the input image size
        if (attn_mask.shape[2:3] != image_size):
            attn_mask = resize_tensor(attn_mask, target_size=image_size)

        # Subtract the final logits from the attention mask.
        remaining_attn: torch.Tensor = (attn_mask - net_logits).clamp(min=0.0, max=1.0)

        # Squeeze the channel dimension.
        net_logits = net_logits.permute(1,0,2,3).squeeze(0)
        _, height, width = net_logits.shape

        context.util.signal_progress("Finalizing mask", 0.9)
        image_out = tensor_to_pil(net_logits, mode="L")
        _metadata = MetadataField({
            "origin": self.image.image_name,
            "segmentation_model": self.model_type,
            "use_tiling": self.use_tiling,
            "smoothing": self.smoothing,
            "min_threshold": self.min_threshold,
            "negative_strength": self.negative_strength,
            "p_blend_mode": self.p_blend_mode,
            "prompt_positive": self.prompt_positive,
            "n_blend_mode": self.n_blend_mode,
            "prompt_negative": self.prompt_negative,
            "compare_mode": self.compare_mode,
            "final_contrast": self.final_contrast,
            "confidence_threshold": self.confidence_threshold,
        })
        mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK, metadata=_metadata)
        context.util.signal_progress("Finished", 1)
        return AdvancedMaskOutput(
            mask=MaskingField.build(
                context=context,
                tensor=net_logits,
                mode=EMaskingMode.GRADIENT,
                metadata=_metadata
            ),
            remaining_attention=MaskingField.build(
                context=context,
                tensor=remaining_attn,
                mode=EMaskingMode.GRADIENT,
            ),
            image=ImageField(image_name=mask_dto.image_name),
            width=width,
            height=height,
        )
