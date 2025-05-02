
import torch
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.services.shared.invocation_context import (
    ImageCategory,
    InvocationContext,
    WithBoard,
)
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor as pil_to_tensor

from siscos_nodes.src.siscos_nodes.masking.enums import EMaskingMode
from siscos_nodes.src.siscos_nodes.util.primitives import (
    MaskingField,
)


@invocation_output('mask_image_output')
class MaskImageNodeOutput(BaseInvocationOutput):
    image: ImageField = InputField(title="Image")

@invocation(
    "mask_image",
    title="Image Mask",
    tags=["mask", "filter"],
    category="mask",
    version="0.0.2",
)
class MaskImageInvocation(BaseInvocation, WithBoard):
    mask: MaskingField = InputField(title="Mask")
    image: ImageField = InputField(title="Image")
    invert: bool = InputField(
        title="Invert",
        default=False,
        description="Invert the mask before applying it to the image.",
    )

    def invoke(self, context: InvocationContext) -> MaskImageNodeOutput:
        mask_in = self.mask.load(context)
        image_in = context.images.get_pil(self.image.image_name)
        # Ensure the image tensor is on the same device as the mask
        image_tensor: torch.Tensor = pil_to_tensor(image_in).to(mask_in.device)
        # Match the mask size to the image size
        if (mask_in.shape[-2:] != image_tensor.shape[-2:]):
            mask_in = torch.nn.functional.interpolate(
                mask_in.unsqueeze(0),
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Invert the mask if required
        if (self.invert):
            mask_in = 1 - mask_in

        result: torch.Tensor
        # Switch behaviour based on the image mode
        match (image_in.mode):
            case "L", "1", "I", "F":
                result = image_tensor * mask_in
            case "RGB":
                assert self.mask.mode != EMaskingMode.IMAGE_COMPOUND, "Mask mode IMAGE_COMPOUND is not supported for RGB images"
                result = image_tensor * mask_in
            case "RGBA":
                assert self.mask.mode != EMaskingMode.IMAGE_COMPOUND, "Mask mode IMAGE_COMPOUND is not supported for RGBA images"
                # Only apply mask to the Alpha channel
                result = image_tensor.clone()
                result[3:4, :, :] *= mask_in
            case _:
                raise ValueError(f"Unsupported image mode: {image_in.mode} for mask mode: {self.mask.mode}")

        # Save the result
        image_out = to_pil_image(result)
        result_dto = context.images.save(image=image_out, image_category=ImageCategory.GENERAL)

        return MaskImageNodeOutput(
            image=ImageField(image_name=result_dto.image_name),
        )
