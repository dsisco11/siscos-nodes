from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, GroupViTModel

from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.raw_model import RawModel
from invokeai.backend.util.devices import TorchDevice

from ..tensor_common import print_tensor_stats, scale_logits
from .segmentation_model import SegmentationModel


class GroupVitPipeline(RawModel):
    """A wrapper class for the transformers GroupViT model and processor that makes it compatible with the model manager."""

    _model: GroupViTModel
    _processor: AutoProcessor

    def __init__(self, model: GroupViTModel, processor: AutoProcessor):

        assert isinstance(model, GroupViTModel), f"Model {model} is not a GroupViT model."
        self._model = model
        self._processor = processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self._model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data

        return calc_model_size_by_data(model=self._model, logger=None)

    def run(
        self,
        image: Image.Image,
        prompts: list[str] | None,
    ) -> torch.Tensor:
        """Segment the image using the GroupViT model.

        Args:
            image (Image.Image): The input image to segment.
            prompts (list[str] | None): The prompts to use for segmentation. If None, uses the default prompts.

        Returns:
            torch.Tensor: The segmented image as a tensor.
        """
        if prompts is None or len(prompts) == 0:
            raise ValueError("No prompts provided for segmentation.")

        # Add a single "empty" prompt to capture ambient noise
        prompts.insert(0, "")

        inputs = self._processor(
            text=prompts,
            images=[image],# repeat the image for each prompt
            return_tensors="pt",
            padding="max_length",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_segmentation=True)

        # outputs.segmentation_logits is a tensor of shape (batch_size, num_prompts, height, width)
        scale = self._model.logit_scale.exp()
        # logits: torch.Tensor = scale_logits(outputs.segmentation_logits, scale)
        noise_logits: torch.Tensor = outputs.segmentation_logits[:, 0:1, :, :]
        raw_logits: torch.Tensor = outputs.segmentation_logits[:, 1:, :, :]
        logits: torch.Tensor = torch.sub(raw_logits, noise_logits).div(scale)
        print("==========================================================")
        # print("Input Prompt:", prompts)
        # print("Input Tokens:", inputs["input_ids"])
        print_tensor_stats(logits, "Logits")
        print("==========================================================")
        return logits

class GroupVitSegmentationModel(SegmentationModel):
    @staticmethod
    def _load(model_path: Path) -> GroupVitPipeline:
        _model = GroupViTModel.from_pretrained(model_path,
            local_files_only=True,
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        assert isinstance(_model, GroupViTModel), "Model is not a GroupViT model."

        _processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        return GroupVitPipeline(model=_model, processor=_processor)

    def execute(self, context: InvocationContext, image_in: Image.Image, prompts: list[str] | None) -> torch.Tensor: # (B, total_prompts, H₁, W₁)
        """Run the model on the given image and prompts.

        Args:
            image (Image.Image): The input image.
            prompts (list[str] | None): The prompts to use for segmentation.

        Returns:
            torch.Tensor: The output tensor from the model.
        """

        pipeline: GroupVitPipeline
        with (
            context.models.load_remote_model(
                source="nvidia/groupvit-gcc-yfcc", loader=GroupVitSegmentationModel._load
            ) as pipeline, # type: ignore
        ):
            return pipeline.run(image_in, prompts=prompts)

# class MaskingNode():
#     def apply_feathering(self, tensor_in, radius):
#         """Apply an expansion to the mask tensor."""
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
#         tensor_in = numpy.array(tensor_in)
#         tensor_in = cv2.dilate(tensor_in, kernel, iterations=1)
#         tensor_in = torch.from_numpy(tensor_in).float()
#         tensor_in = tensor_in.unsqueeze(0).unsqueeze(0)
#         tensor_in = tensor_in.to(torch.float32)
#         return tensor_in

# @invocation(
#     "segmentation_groupvit_resolver",
#     title="Mask Resolver (GroupViT)",
#     tags=["mask", "segmentation", "groupvit"],
#     category="mask",
#     version="0.0.1",
# )
# class ResolveMaskGroupvitInvocation(BaseInvocation, MaskingNode, WithBoard):
#     """Uses the GroupViT model to resolve a mask from the given image using a positive & negative prompt.

#     Returns a mask which matches the positive prompt and does not match the negative prompt.
#     The mask is created by thresholding the output of the GroupViT model, which is a probability map of the image.

# """

#     image: ImageField = InputField(description="")
#     prompt_positive: list[str] = InputField(description="", )
#     prompt_negative: list[str] = InputField(description="")
#     smoothing: float = InputField(default=8.0, description="")
#     min_threshold: float = InputField(
#         default=0.5, description="Minimum certainty for the mask, values below this will be clipped to 0."
#     )
#     mask_feathering: float = InputField(
#         default=0, description="Feathering radius for the mask. 0 means no feathering."
#     )
#     mask_blur: float = InputField(default=4.0, description="Blur radius for the mask. 0 means no blur.")


#     def invoke(self, context: InvocationContext) -> AdvancedMaskOutput:
#         image_in = context.images.get_pil(self.image.image_name, mode="RGB")
#         image_size = image_in.size
#         pos_prompt_count = len(self.prompt_positive)
#         neg_prompt_count = len(self.prompt_negative)
#         model_id: GroupVitModelKey = "groupvit-default"

#         # If we have no prompts, we can skip the model call and just return a blank mask.
#         # if self.prompt_positive is None or len(self.prompt_positive) == 0:
#         #     mask_tensor = torch.ones((image_size[1], image_size[0]), dtype=torch.bool)
#         #     mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
#         #     mask_tensor_name = context.tensors.save(mask_tensor)
#         #     return AdvancedMaskOutput(
#         #         mask=TensorField(tensor_name=mask_tensor_name),
#         #         width=image_size[0],
#         #         height=image_size[1],
#         #     )

#         with (
#             context.models.load_remote_model(
#                 source=GROUPVIT_MODEL_IDS[model_id], loader=ResolveMaskGroupvitInvocation._load_model
#             ) as pipeline,
#         ):
#             assert isinstance(pipeline, GroupVitPipeline), f"Pipeline {pipeline} is not a GroupViTPipeline."
#             logits = pipeline.segment(image_in, prompts=self.prompt_positive + self.prompt_negative) # (B, num_labels, H₁, W₁)

#         pos_logits = logits[:, :pos_prompt_count].mean(1, keepdim=True)   # (B, 1, H₁, W₁)
#         net_logits = pos_logits

#         if (neg_prompt_count > 0):
#             neg_logits = logits[:, pos_prompt_count:].mean(1, keepdim=True)  # (B, 1, H₁, W₁)
#             net_logits = (pos_logits - neg_logits).clamp(min=0, max=1)

#         # Normalize the values to be between 0 and 1
#         # net_logits = normalize_tensor(net_logits)

#         # if 0 < self.min_threshold:
#             # mask_tensor = mask_tensor - self.min_threshold

#         if 0 < self.smoothing:
#             net_logits = gaussian_blur(net_logits, sigma=self.smoothing)

#         # Upscale the mask tensor to the original image size
#         net_logits = upscale_tensor(net_logits, target_size=image_size)

#         if 0 < self.mask_feathering:
#             net_logits = self.apply_feathering(net_logits, self.mask_feathering)

#         if 0 < self.mask_blur:
#             net_logits = gaussian_blur(net_logits, sigma=self.mask_blur)

#         # Squeeze the channel dimension.
#         net_logits = net_logits.squeeze(0)
#         _, height, width = net_logits.shape

#         image_out = tensor_to_image(net_logits, mode="L")
#         mask_dto = context.images.save(image_out, image_category=ImageCategory.MASK)
#         return AdvancedMaskOutput(
#             mask=MaskField(asset_id=mask_dto.image_name, mode="image_luminance"),
#             image=ImageField(image_name=mask_dto.image_name),
#             width=width,
#             height=height,
#         )
