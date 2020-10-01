from box_embeddings.parameterizations.box_tensor import BoxTensor
from .pooling import BoxPooler
import torch


def hard_intersection_pooler(
    boxes: BoxTensor,
    mask: torch.BoolTensor = None,
    dim: int = 0,
    keepdim: bool = False,
) -> BoxTensor:
    box_z = boxes.z
    box_Z = boxes.Z

    if mask is not None:
        box_z[mask] -= float("inf")
        box_Z[mask] += float("inf")
    z = torch.max(box_z, dim=dim, keepdim=keepdim)[0]
    Z = torch.min(box_Z, dim=dim, keepdim=keepdim)[0]

    return boxes.like_this_from_zZ(z, Z)


@BoxPooler.register("hard-intersection")
class HardIntersectionBoxPooler(BoxPooler):

    """Pools a box tensor using hard intersection operation"""

    def __init__(self, dim: int = 0, keepdim: bool = False):
        super().__init__()  # type:ignore
        self.dim = dim
        self.keepdim = keepdim

    def forward(  # type:ignore
        self, box_tensor: BoxTensor, mask: torch.BoolTensor = None
    ) -> BoxTensor:
        return hard_intersection_pooler(
            box_tensor, mask=mask, dim=self.dim, keepdim=self.keepdim
        )
