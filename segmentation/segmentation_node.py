from typing import Literal

import cv2
import numpy
import torch
from torchvision.transforms.functional import to_pil_image as tensor_to_image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.invocation_api import WithBoard

from ..primitives import AdvancedMaskOutput, MaskField
from ..tensor_common import (
    apply_feathering_ellipse,
    gaussian_blur,
    normalize_tensor,
    print_tensor_stats,
    upscale_tensor,
)
from .clipseg import CLIPSegSegmentationModel
from .groupvit import GroupVitSegmentationModel
from .segmentation_model import SegmentationModel

ImageSegmentationModelType = Literal["clipseg", "groupvit"]
SEGMENTATION_MODEL_TYPES: dict[ImageSegmentationModelType, type[SegmentationModel]] = {
    "clipseg": CLIPSegSegmentationModel,
    "groupvit": GroupVitSegmentationModel,
}

@invocation(
    "segmentation_mask_resolver",
    title="Segmentation Mask Resolver",
    tags=["mask", "segmentation", "txt2mask"],
    category="mask",
    version="0.0.1",
)
class ResolveSegmentationMaskInvocation(BaseInvocation, WithBoard):
    """Uses the chosen image-segmentation model to resolve a mask from the given image using a positive & negative prompt.

    Returns a mask which matches the positive prompt(s) and does not match the negative prompt(s).
    The resulting mask indicates the relative intensity of how strongly different areas of the image match the positive prompt(s) minus the negative prompt(s).

"""

    image: ImageField = InputField(description="")
    model_type: ImageSegmentationModelType = InputField(
        default="clipseg", description="The model to use for segmentation."
    )
    prompt_positive: list[str] = InputField(description="")
    prompt_negative: list[str] = InputField(description="")
    smoothing: float = InputField(default=4.0, description="")
    min_threshold: float = InputField(
        default=0.0, description="Minimum certainty for the mask, values below this will be clipped to 0."
    )
    mask_feathering: float = InputField(
        default=2, description="Feathering radius for the mask. 0 means no feathering."
    )
    mask_blur: float = InputField(default=2.0, description="Blur radius for the mask. 0 means no blur.")

    def invoke(self, context: InvocationContext) -> AdvancedMaskOutput:
        image_in = context.images.get_pil(self.image.image_name, mode="RGB")
        image_size = image_in.size
        pos_prompt_count = len(self.prompt_positive)
        neg_prompt_count = len(self.prompt_negative)
        model: SegmentationModel = SEGMENTATION_MODEL_TYPES[self.model_type]()

        # If we have no prompts, we can skip the model call and just return a blank mask.
        # if (pos_prompt_count == 0 and neg_prompt_count == 0):
        #     mask_tensor = torch.ones((image_size[1], image_size[0]), dtype=torch.bool)
        #     mask_out = tensor_to_image(mask_tensor, mode="L")
        #     mask_dto:ImageDTO = context.images.save(image=mask_out, image_category=ImageCategory.MASK)
        #     return AdvancedMaskOutput(
        #         mask=MaskField(asset_id=mask_dto.image_name, mode="boolean"),
        #         image=ImageField(image_name=mask_dto.image_name),
        #         width=image_size[0],
        #         height=image_size[1],
        #     )

        if (pos_prompt_count == 0):
            # Done to test if the model is working correctly
            # add a blank positive prompt to the list of prompts so we can check if the final mask is zeroed out as expected.
            pos_prompt_count = 1
            self.prompt_positive = [""]

        # add a blank negative prompt to the list of prompts to capture and negate "ambient noise"
        # neg_prompt_count += 1
        # self.prompt_negative.append("")

        context.util.signal_progress("Running model", 0.0)
        _prompts = [x.strip() for x in (self.prompt_positive + self.prompt_negative)]
        logits = model.execute(context, image_in, prompts=_prompts)
        context.util.signal_progress("Processing results", 0.2)

        pos_logits = logits[:, :pos_prompt_count].mean(dim=1, keepdim=True)   # (B, 1, H₁, W₁)
        net_logits = pos_logits

        if (neg_prompt_count > 0):
            neg_logits = logits[:, pos_prompt_count:].mean(dim=1, keepdim=True)  # (B, 1, H₁, W₁)
            net_logits = (pos_logits - neg_logits).clamp(min=0.0)

        context.util.signal_progress("Normalizing results", 0.4)
        # Normalize the values to be between 0 and 1
        # net_logits = net_logits.sigmoid()
        net_logits = net_logits.softmax(dim=2)
        net_logits = normalize_tensor(net_logits)

        # if 0 < self.min_threshold:
            # mask_tensor = mask_tensor - self.min_threshold

        if 0 < self.smoothing:
            context.util.signal_progress("Smoothing results", 0.5)
            net_logits = gaussian_blur(net_logits, sigma=self.smoothing)

        # Upscale the mask tensor to the original image size
        context.util.signal_progress("Upscaling mask", 0.6)
        net_logits = upscale_tensor(net_logits, target_size=image_size)

        if 0 < self.mask_feathering:
            context.util.signal_progress("Feathering mask", 0.7)
            net_logits = apply_feathering_ellipse(net_logits, self.mask_feathering)

        if 0 < self.mask_blur:
            context.util.signal_progress("Blurring mask", 0.8)
            net_logits = gaussian_blur(net_logits, sigma=self.mask_blur)

        # Squeeze the channel dimension.
        net_logits = neg_logits.softmax(dim=2).squeeze(0)
        print_tensor_stats(net_logits, "Net Logits")
        _, height, width = net_logits.shape

        context.util.signal_progress("Finalizing mask", 0.9)
        image_out = tensor_to_image(net_logits, mode="L")
        mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK)
        return AdvancedMaskOutput(
            mask=MaskField(asset_id=mask_dto.image_name, mode="image_luminance"),
            image=ImageField(image_name=mask_dto.image_name),
            width=width,
            height=height,
        )
