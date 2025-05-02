import torch
from invokeai.app.invocations.baseinvocation import (
    BaseInvocationOutput,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, OutputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    MetadataField,
)
from invokeai.backend.util.devices import TorchDevice
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_pil_image as tensor_to_pil
from torchvision.transforms.functional import to_tensor as image_to_tensor

from siscos_nodes.src.siscos_nodes.masking.enums import EMaskingMode
from siscos_nodes.src.siscos_nodes.util.tensor_common import (
    MaskTensor,
)


class MaskingField(BaseModel):
    """A masking primitive field."""

    asset_id: str = Field(description="The id/name of the mask image within the asset cache system.")
    mode: EMaskingMode = Field(description="The masking mode specifies how the mask is represented.")

    def __init__(self, mode: EMaskingMode | None = None, asset_id: str | None = None, image_name: str | None = None, tensor_name: str | None = None):
        """Initialize the MaskingField with an asset ID and mode."""
        if (asset_id is None and image_name is None and tensor_name is None):
            raise ValueError("Either asset_id, image_name or tensor_name must be provided.")
        
        # Use the first non-None value
        _id: str = asset_id or image_name or tensor_name # type: ignore
        _mode: EMaskingMode = mode or (EMaskingMode.IMAGE_LUMINANCE if image_name else EMaskingMode.BOOLEAN) # type: ignore
        if not _id:
            raise ValueError("Either asset_id, image_name or tensor_name must be provided.")
        
        super().__init__(asset_id=_id, mode=_mode)

    def load(self, context: InvocationContext) -> torch.Tensor: # [C, H, W]
        """Load the mask from the asset cache."""
        device: torch.device = TorchDevice.choose_torch_device()
        match (self.mode):
            case EMaskingMode.BOOLEAN:
                return image_to_tensor(context.images.get_pil(self.asset_id)).to(device=device)
            case EMaskingMode.GRADIENT:
                return image_to_tensor(context.images.get_pil(self.asset_id)).to(device=device)
            case EMaskingMode.IMAGE_LUMINANCE:
                return image_to_tensor(context.images.get_pil(self.asset_id)).to(device=device)
            case EMaskingMode.IMAGE_ALPHA:
                return image_to_tensor(context.images.get_pil(self.asset_id)).split(1)[-1].to(device=device)
            case EMaskingMode.IMAGE_COMPOUND:
                return image_to_tensor(context.images.get_pil(self.asset_id)).to(device=device)
            case _:
                raise ValueError(f"Unsupported mask mode: {self.mode}")
            
    @classmethod
    def build(cls, context: InvocationContext, tensor: torch.Tensor, mode: EMaskingMode, metadata: MetadataField | None = None) -> "MaskingField":
        """Build a MaskingField from a tensor."""
        assert tensor is not None, "Tensor must not be None"
        if (tensor.dim() == 3):
            assert tensor.shape[0] == 1, f"Tensor must have shape [1, H, W], but got {tensor.shape}"
            tensor = tensor.unsqueeze(0)
        elif (tensor.dim() == 4):
            if (tensor.shape[0] > 1 or tensor.shape[1] > 1):
                raise ValueError(f"Unsupported mask shape: {tensor.shape}")
            else:
                tensor = tensor.squeeze(0).squeeze(0)
        
        pil_mode = MaskTensor.getPILMode(mode)
        formatted_tensor = MaskTensor.format(tensor, mode)
        image = tensor_to_pil(formatted_tensor, mode=pil_mode)
        dto = context.images.save(image, image_category=ImageCategory.MASK, metadata=metadata)
        if (dto is None):
            raise ValueError("Failed to save image to asset cache.")
        return MaskingField(asset_id=dto.image_name, mode=mode)




@invocation_output("masking_node_output")
class MaskingNodeOutput(BaseInvocationOutput):
    mask: MaskingField = OutputField(title="Mask")

@invocation_output("adv_mask_output")
class AdvancedMaskOutput(BaseInvocationOutput):

    mask: MaskingField = OutputField(title="Mask", description="The mask.")
    remaining_attention: MaskingField = OutputField(title="Remaining Attention", description="The initial attention mask excluding the returned mask.")
    image: ImageField = OutputField(description="The mask as an image.")
    width: int = OutputField(description="The width of the mask in pixels.")
    height: int = OutputField(description="The height of the mask in pixels.")
