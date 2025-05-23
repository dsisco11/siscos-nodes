from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.raw_model import RawModel

from ..tensor_common import print_tensor_stats
from .segmentation_model import SegmentationModel


class CLIPSegPipeline(RawModel):
    """A wrapper class for the transformers CLiPSeg model and processor that makes it compatible with the model manager."""

    _model: CLIPSegForImageSegmentation
    _processor: CLIPSegProcessor

    def __init__(self, model: CLIPSegForImageSegmentation, processor: CLIPSegProcessor):

        assert isinstance(model, CLIPSegForImageSegmentation), f"Model {model} is not a CLiPSeg model."
        self._model = model
        self._processor = processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self._model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._model)

    def run(
        self,
        image: Image.Image,
        prompts: list[str] | None,
    ) -> torch.Tensor:
        """Segment the image using the CLiPSeg model.

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
            images=[image] * len(prompts),# repeat the image for each prompt
            return_tensors="pt",
            padding="max_length",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        noise_logits: torch.Tensor = outputs.logits[0:1, :, :]
        raw_logits: torch.Tensor = outputs.logits[1:, :, :]
        logits: torch.Tensor = torch.sub(raw_logits, noise_logits).unsqueeze(0)  # (B, num_labels, H₁, W₁)
        return logits

class CLIPSegSegmentationModel(SegmentationModel):
    @staticmethod
    def _load_from_path(model_path: Path) -> CLIPSegPipeline:
        _model = CLIPSegForImageSegmentation.from_pretrained(model_path,
            local_files_only=True,
            # TODO:(sisco): Setting the torch_dtype here doesn't work. It causes the model to complain that the imput tensors aren't of the same type.
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        assert isinstance(_model, CLIPSegForImageSegmentation), "Model is not a CLiPSeg model."

        _processor = CLIPSegProcessor.from_pretrained(model_path, local_files_only=True,
            # TODO:(sisco): Setting the torch_dtype here doesn't work. It causes the model to complain that the imput tensors aren't of the same type.
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        return CLIPSegPipeline(model=_model, processor=_processor)

    def execute(self, context: InvocationContext, image_in: Image.Image, prompts: list[str] | None) -> torch.Tensor: # (B, total_prompts, H₁, W₁)
        """Run the model on the given image and prompts.

        Args:
            image (Image.Image): The input image.
            prompts (list[str] | None): The prompts to use for segmentation.

        Returns:
            torch.Tensor: The output tensor from the model.
        """

        pipeline: CLIPSegPipeline
        with (
            context.models.load_remote_model(
                source="CIDAS/clipseg-rd64-refined", loader=CLIPSegSegmentationModel._load_from_path
            ) as pipeline, # type: ignore
        ):
            return pipeline.run(image_in, prompts=prompts)
