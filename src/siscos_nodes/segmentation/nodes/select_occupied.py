from typing import TypeAlias

import torch
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    invocation,
)
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.primitives import (
    BoundingBoxCollectionOutput,
    BoundingBoxField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext

from siscos_nodes.src.siscos_nodes.util.primitives import MaskingField

ListOfBoundingBoxes: TypeAlias = list[list[int]]
"""A list of bounding boxes. Each bounding box is in the format [xmin, ymin, xmax, ymax]."""

@invocation(
    "select_occupied",
    title="Select Occupied",
    tags=["segmentation", "identification"],
    category="segmentation",
    version="0.0.1",
)
class SelectOccupiedInvocation(BaseInvocation):
    """Outputs the bounding boxes of the occupied regions in the input mask.
    """

    mask: MaskingField = InputField(title="Mask")

    def invoke(self, context: InvocationContext) -> BoundingBoxCollectionOutput:
        mask_in = self.mask.load(context)

        results: torch.Tensor = SelectOccupiedInvocation.resolve_bounding_boxes(tensor=mask_in.squeeze(dim=0)).cpu()
        # transform the results to a list[int[4]] of bounding boxes]
        detected: ListOfBoundingBoxes = results.tolist()

        # Convert detections to BoundingBoxCollectionOutput.
        bounding_boxes: list[BoundingBoxField] = []
        for box in detected:
            bounding_boxes.append(
                BoundingBoxField(
                    x_min=box[0],
                    y_min=box[1],
                    x_max=box[2],
                    y_max=box[3],
                    score=1.0,  # Placeholder score, as we don't have a score for these boxes
                )
            )

        return BoundingBoxCollectionOutput(collection=bounding_boxes)
    
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def resolve_bounding_boxes(tensor: torch.Tensor, num_iters: int = 10000) -> torch.Tensor:
        """
        Find bounding boxes of >0 connected components in a 2D tensor.

        Args:
            tensor    (H,W float): input map, on CUDA device
            num_iters (int): max propagation iterations (usually << H+W)

        Returns:
            boxes (N,4 int64): for each component, [y_min, x_min, y_max, x_max], on the same device
        """
        # --- 1) setup & binary mask ---
        assert tensor.dim() == 2, "Input must be 2D"
        device = tensor.device
        H, W = tensor.shape
        mask = tensor > 0

        # early exit
        if not mask.any():
            return torch.empty((0, 4), device=device, dtype=torch.int64)

        # --- 2) seed each positive pixel with a unique float label ID ---
        idx = torch.arange(1, H * W + 1, device=device, dtype=torch.float32)
        idx = idx.view(1, 1, H, W)
        labels = idx * mask.unsqueeze(0).unsqueeze(0).float()

        # --- 3) propagate labels by 3×3 max‑pool until convergence ---
        for _ in range(num_iters):
            dilated = torch.nn.functional.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
            new_labels = torch.where(mask.unsqueeze(0).unsqueeze(0), dilated, torch.zeros_like(labels))
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
        labels = labels.view(H, W).long()  # now each component has a constant int label

        # --- 4) flatten and isolate only positive pixels ---
        flat_labels = labels.view(-1) - 1       # now 0…(N-1), with background = -1
        flat_mask   = mask.view(-1)

        fg_labels = flat_labels[flat_mask]
        # —— use functional unique & sort ——
        uniq      = torch.unique(fg_labels)
        uniq_sorted, _ = torch.sort(uniq)
        N = uniq_sorted.size(0)

        # 5) map every fg pixel to its component index
        comp_idx = torch.searchsorted(uniq_sorted, fg_labels)

        # 6) coords for every fg pixel
        rows = torch.arange(H, device=tensor.device).view(H, 1).expand(H, W)
        cols = torch.arange(W, device=tensor.device).view(1, W).expand(H, W)
        row_flat = rows.reshape(-1)[flat_mask]
        col_flat = cols.reshape(-1)[flat_mask]

        # --- 7) scatter_reduce to get mins & maxes per component ---
        y_min = torch.full((N,), H,  device=device, dtype=torch.int64)
        y_max = torch.full((N,), -1, device=device, dtype=torch.int64)
        x_min = torch.full((N,), W,  device=device, dtype=torch.int64)
        x_max = torch.full((N,), -1, device=device, dtype=torch.int64)

        y_min.scatter_reduce_(0, comp_idx, row_flat, reduce="amin", include_self=True)
        y_max.scatter_reduce_(0, comp_idx, row_flat, reduce="amax", include_self=True)
        x_min.scatter_reduce_(0, comp_idx, col_flat, reduce="amin", include_self=True)
        x_max.scatter_reduce_(0, comp_idx, col_flat, reduce="amax", include_self=True)

        # --- 8) stack into (N,4) box tensor and return ---
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)
