from typing import Any
import torch
from box_embeddings.common.registrable import Registrable
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.utils import tiny_value_of_dtype

eps = tiny_value_of_dtype(torch.float)


class Volume(torch.nn.Module, Registrable):

    """Base volume class"""

    default_implementation = "hard"

    def __init__(self, log_scale: bool = True, **kwargs: Any) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.log_scale = log_scale

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Raises:
            NotImplementedError: base class
        """
        raise NotImplementedError


def hard_volume(box_tensor: BoxTensor) -> torch.Tensor:
    """Volume of boxes. Returns 0 where boxes are flipped.

    Args:
        box_tensor: input

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Examples:
        >>> from box_embeddings.parameterizations.box_tensor import BoxTensor
        >>> from box_embeddings.modules.volume import hard_volume
        >>> z = [-2.0]*100
        >>> Z = [0.0]*100
        >>> input = [z, Z]
        >>> box1 = BoxTensor(torch.tensor(input))
        >>> hard_volume(box1) # doctest: +NORMALIZE_WHITESPACE
        tensor(1.2677e+30)
    """

    return torch.prod((box_tensor.Z - box_tensor.z).clamp_min(0), dim=-1)


def log_hard_volume(box_tensor: BoxTensor) -> torch.Tensor:
    """
    Logged hard volume of box.

    Args:
        box_tensor: input

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Examples:
        >>> from box_embeddings.parameterizations.box_tensor import BoxTensor
        >>> from box_embeddings.modules.volume import hard_volume
        >>> z = [-2.0]*100
        >>> Z = [0.0]*100
        >>> input = [z, Z]
        >>> box1 = BoxTensor(torch.tensor(input))
        >>> log_hard_volume(box1) # doctest: +NORMALIZE_WHITESPACE
        tensor(69.3147)
    """
    res = torch.sum(
        torch.log((box_tensor.Z - box_tensor.z).clamp_min(eps)), dim=-1
    )

    return res


@Volume.register("hard")
class HardVolume(Volume):

    """Hard ReLU based volume."""

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Hard ReLU base volume.

        Args:
            box_tensor: input

        Returns:
            torch.Tensor

        """

        if self.log_scale:
            return log_hard_volume(box_tensor)
        return hard_volume(box_tensor)
