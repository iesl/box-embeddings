import torch

from box_embeddings.common.utils import tiny_value_of_dtype
from box_embeddings.modules.volume._volume import _Volume
from box_embeddings.parameterizations import BoxTensor

eps = tiny_value_of_dtype(torch.float)


def hard_volume(box_tensor: BoxTensor, log_scale: bool = True) -> torch.Tensor:
    """Volume of boxes. Returns 0 where boxes are flipped.

    Args:
        box_tensor: input
        log_scale: Whether the output should be in log scale or not.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)
    """

    if log_scale:
        return torch.sum(
            torch.log((box_tensor.Z - box_tensor.z).clamp_min(eps)), dim=-1
        )

    return torch.prod((box_tensor.Z - box_tensor.z).clamp_min(0), dim=-1)


@_Volume.register("hard")
class HardVolume(_Volume):

    """Hard ReLU based volume."""

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Hard ReLU base volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor

        """
        return hard_volume(box_tensor, self.log_scale)
