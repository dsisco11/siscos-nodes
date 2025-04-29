from pathlib import Path
from typing import Optional, Union

import torch
from invokeai.app.services.model_load.model_load_base import (
    LoadedModelWithoutConfig,
    ModelLoadServiceBase,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import (
    ModelLoaderRegistry,
)
from invokeai.backend.model_manager.load.model_util import calc_module_size
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.raw_model import RawModel
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from ..util.tensor_common import normalize_logits
from .segmentation_model import ISegmentationPipeline, SegmentationModel


class CLIPSegRawModel(RawModel):
    """A wrapper class for the transformers CLiPSeg model that makes it compatible with the model manager."""

    _model: CLIPSegForImageSegmentation
    _processor: CLIPSegProcessor

    def __init__(self, model: CLIPSegForImageSegmentation, processor: CLIPSegProcessor):
        self._model = model
        self._processor = processor

    @property
    def model(self) -> CLIPSegForImageSegmentation:
        return self._model
    
    @property
    def processor(self) -> CLIPSegProcessor:
        return self._processor
    
    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> AnyModel:
        """Load the model from a checkpoint file.

        Args:
            file_path (Path): The path to the model checkpoint file.
            device (Optional[torch.device]): The device to load the model on.
            dtype (Optional[torch.dtype]): The data type to use for the model.

        Returns:
            CLIPSegRawModel: An instance of the CLIPSegRawModel class.
        """
        _model = CLIPSegForImageSegmentation.from_pretrained(file_path, local_files_only=True
            # TODO:(sisco): Setting the torch_dtype here doesn't work. It causes the model to complain that the input tensors aren't of the same type.
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        _processor = CLIPSegProcessor.from_pretrained(file_path, local_files_only=True
            # TODO:(sisco): Setting the torch_dtype here doesn't work. It causes the model to complain that the input tensors aren't of the same type.
            # torch_dtype=TorchDevice.choose_torch_dtype()
        )
        return cls(model=_model, processor=_processor)
    
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:        
        self._model.to(device=device, dtype=dtype)

        
    def calc_size(self) -> int:
        return calc_module_size(self._model)
    

# @ModelLoaderRegistry.register(
#     base=BaseModelType.Any,
#     type='',
#     format=ModelFormat.EmbeddingFile,
# )
class CLIPSegModelLoader(ModelLoader):
    """Class to load CLiPSeg models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("There are no submodels in a CLiPSeg model.")
        model = CLIPSegRawModel.from_checkpoint(
            file_path=config.path,
            dtype=self._torch_dtype,
        )
        return model
    


class CLIPSegSegmentationPipeline(LoadedModelWithoutConfig, ISegmentationPipeline):
    """A wrapper class for the transformers CLiPSeg model that makes it compatible with the model manager."""

    _instance: CLIPSegRawModel

    def __init__(self, model: CLIPSegRawModel):
        self._instance = model
    
    def execute(
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

        inputs = self._instance.processor(
            text=prompts,
            images=[image] * len(prompts),# repeat the image for each prompt
            return_tensors="pt",
            padding="max_length",
        ).to(device=self._instance.model.device)

        with torch.no_grad():
            outputs = self._instance.model(**inputs)

        noise_logits: torch.Tensor = outputs.logits[0:1, :, :]
        raw_logits: torch.Tensor = outputs.logits[1:, :, :]
        logits: torch.Tensor = torch.sub(raw_logits, noise_logits)# (num_prompts, H₁, W₁)

        # modify tensor to match shape: (num_prompts, 1, H₁, W₁)
        return normalize_logits(logits).unsqueeze(1)

class CLIPSegSegmentationModel(SegmentationModel):
    def execute(self, context: InvocationContext, image_in: Image.Image, prompts: list[str] | None) -> torch.Tensor: # (total_prompts, 1, H₁, W₁)
        """Run the model on the given image and prompts.

        Args:
            image (Image.Image): The input image.
            prompts (list[str] | None): The prompts to use for segmentation.

        Returns:
             Tensor<total_prompts, 1, H₁, W₁>: The output tensor from the model.
        """

        rawModel: CLIPSegRawModel
        with (
            context.models.load_remote_model(
                source="CIDAS/clipseg-rd64-refined", loader=CLIPSegRawModel.from_checkpoint
            ) as rawModel, # type: ignore 
        ):
            pipeline = CLIPSegSegmentationPipeline(rawModel)
            return pipeline.execute(image_in, prompts=prompts)
