from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations.box_tensor import BoxTensor
from .pooling import BoxPooler
import torch


def bag_of_boxes_pooler(
    boxes: BoxTensor,
    mask: Optional[torch.BoolTensor] = None,
    weights: Optional[torch.Tensor] = None,
    dim: int = 0,
    keepdim: bool = False,
) -> BoxTensor:
    box_z = boxes.z
    box_Z = boxes.Z

    if weights is None:
        weights = torch.ones_like(box_z)

    if mask is not None:
        weights = weights * mask
    denominator = torch.sum(weights, dim=dim, keepdim=keepdim)
    z = torch.sum(box_z * weights, dim=dim, keepdim=keepdim) / (denominator + 1e-14)
    Z = torch.sum(box_Z * weights, dim=dim, keepdim=keepdim) / (denominator + 1e-14)

    return boxes.like_this_from_zZ(z, Z)


@BoxPooler.register("bag-of-boxes-pooler")
class BagOfBoxesBoxPooler(BoxPooler):

    """Pools a box tensor using hard intersection operation"""

    def __init__(self, dim: int = 0, keepdim: bool = False):
        super().__init__()  # type:ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(  # type:ignore
        self,
        box_tensor: BoxTensor,
        mask: torch.BoolTensor = None,
        weights: torch.BoolTensor = None,
    ) -> BoxTensor:  # type:ignore
        """

        Args:
            box_tensor: Input
            mask: With shape as box_tensor.box_shape[:-1].
                0 at a position means mask it.
            weights:  With shape as box_tensor.box_shape[:-1].

        Returns:
            BoxTensor: Pooled output
        """

        return bag_of_boxes_pooler(
            box_tensor,
            mask=mask,
            weights=weights,
            dim=self.dim,
            keepdim=self.keepdim,
        )
