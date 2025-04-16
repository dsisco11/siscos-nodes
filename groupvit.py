import cv2
import torch
import numpy
from typing import Literal, Optional
from PIL import Image, ImageFilter
import torch.nn.functional as TorchFunctional
from transformers import AutoModelForMaskGeneration, AutoProcessor, GroupViTModel
from torchvision.transforms.functional import to_pil_image as tensor_to_image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, TensorField
from invokeai.app.invocations.primitives import MaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.raw_model import RawModel

GroupVitModelKey = Literal["groupvit-default"]
GROUPVIT_MODEL_IDS: dict[GroupVitModelKey, str] = {
    "groupvit-default": "nvidia/groupvit-gcc-yfcc",
}

def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a 4D torch tensor (batch, channels, height, width).

    Args:
        tensor (torch.Tensor): Input tensor to blur.
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Blurred tensor.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Create 2D Gaussian kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)

    # Expand kernel to match input channels
    kernel = kernel.to(tensor.device)
    kernel = kernel.expand(tensor.size(1), 1, kernel_size, kernel_size)

    # Apply Gaussian blur using conv2d
    padding = kernel_size // 2
    blurred_tensor = TorchFunctional.conv2d(tensor, kernel, padding=padding, groups=tensor.size(1))

    return blurred_tensor

class MaskingNode():
    def apply_feathering(self, tensor_in, radius):
        """Apply an expansion to the mask tensor."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        tensor_in = numpy.array(tensor_in)
        tensor_in = cv2.dilate(tensor_in, kernel, iterations=1)
        tensor_in = torch.from_numpy(tensor_in).float()
        tensor_in = tensor_in.unsqueeze(0).unsqueeze(0)
        tensor_in = tensor_in.to(torch.float32)
        return tensor_in

class GroupVitPipeline(RawModel):
    """A wrapper class for the transformers GroupViT model and processor that makes it compatible with the model manager."""

    _model: GroupViTModel
    _processor: AutoProcessor

    def __init__(self, model: AutoModelForMaskGeneration, processor: AutoProcessor):

        assert isinstance(model, GroupViTModel), f"Model {model} is not a GroupViT model."
        self._model = model
        self._processor = processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self._model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._model)
    
    def segment(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
    ) -> torch.Tensor:
        """Segment the image using the GroupViT model.

        Args:
            image (Image.Image): The input image to segment.
            prompts (list[str] | None): The prompts to use for segmentation. If None, uses the default prompts.

        Returns:
            torch.Tensor: The segmented image as a tensor.
        """
        if prompts is None:
            prompts = [""]

        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            output_segmentation=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        return torch.sigmoid(outputs.logits)

@invocation(
    "segmentation_groupvit_resolver",
    title="Mask Resolver (GroupViT)",
    tags=["mask", "segmentation", "groupvit"],
    category="mask",
    version="0.0.1",
)
class ResolveMaskGroupvitInvocation(BaseInvocation, MaskingNode):
    """Uses the GroupViT model to resolve a mask from the given image using a positive & negative prompt.

    Returns a mask which matches the positive prompt and does not match the negative prompt.
    The mask is created by thresholding the output of the GroupViT model, which is a probability map of the image.

"""

    image: ImageField = InputField(description="")
    prompt_positive: list[str] = InputField(description="", )
    prompt_negative: list[str] = InputField(description="")
    smoothing: float = InputField(default=8.0, description="")
    min_threshold: float = InputField(
        default=0.5, description="Minimum certainty for the mask, values below this will be clipped to 0."
    )
    mask_feathering: int = InputField(
        default=0, description="Feathering radius for the mask. 0 means no feathering."
    )
    mask_blur: float = InputField(default=4.0, description="Blur radius for the mask. 0 means no blur.")

    def _load_model(self, model_id: GroupVitModelKey) -> GroupViTModel:
        if model_id not in GROUPVIT_MODEL_IDS:
            raise ValueError(f"Model ID {model_id} not found in GroupViT model IDs.")
        
        _model = AutoModelForMaskGeneration.from_pretrained(GROUPVIT_MODEL_IDS[model_id])        
        assert isinstance(_model, GroupViTModel), f"Model {model_id} is not a GroupViT model."
        
        _processor = AutoProcessor.from_pretrained(GROUPVIT_MODEL_IDS[model_id])
        return GroupVitPipeline(model=_model, processor=_processor)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MaskOutput:
        image_in = context.images.get_pil(self.image.image_name)
        image_size = image_in.size
        mask_tensor: torch.Tensor = None

        pipeline = self._load_model("groupvit-default")

        image_in = image_in.convert("RGB")

        positive_prediction = pipeline.segment(image_in, prompts=self.prompt_positive)
        negative_prediction = pipeline.segment(image_in, prompts=self.prompt_negative)

        prediction = positive_prediction - negative_prediction
        prediction = torch.clamp(prediction, min=self.min_threshold, max=1.0)
        
        mask_tensor = tensor_to_image(prediction, mode="L")
        mask_tensor = mask_tensor.resize(image_size)

        mask_tensor = image_resized_to_grid_as_tensor(mask_tensor, normalize=False)
        mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min())

        if self.smoothing > 0:
            mask_tensor = gaussian_blur(mask_tensor, kernel_size=5, sigma=self.smoothing)

        if self.mask_feathering != 0:
            mask_tensor = self.apply_feathering(mask_tensor, self.mask_feathering)

        if self.mask_blur > 0:
            mask_tensor = gaussian_blur(mask_tensor, kernel_size=5, sigma=self.mask_blur)

        mask_tensor_name = context.tensors.save(mask_tensor)

        return MaskOutput(
            mask=TensorField(tensor_name=mask_tensor_name),
            width=image_size[0],
            height=image_size[1],
        )
