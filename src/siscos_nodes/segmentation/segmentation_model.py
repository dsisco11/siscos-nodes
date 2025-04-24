import torch
from PIL import Image

from invokeai.app.services.shared.invocation_context import InvocationContext


# Abstract base class for generically loading/executing a segmentation model
class SegmentationModel():
    def execute(self, context: InvocationContext, image_in: Image.Image, prompts: list[str] | None) -> torch.Tensor:
        """Run the model on the given image and prompts.

        Args:
            image (Image.Image): The input image.
            prompts (list[str] | None): The prompts to use for segmentation.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
