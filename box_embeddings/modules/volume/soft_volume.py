from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.common.utils import tiny_value_of_dtype
from torch.nn.functional import softplus
import numpy as np

# eps = tiny_value_of_dtype(torch.float)

eps = 1e-23


def soft_volume(
    box_tensor: BoxTensor, beta: float = 1.0, scale: float = 1.0
) -> torch.Tensor:
    """Volume of boxes. Uses softplus instead of ReLU/clamp

    Args:
        box_tensor: input
        beta: the beta parameter for the softplus
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1]

    Example:
        >>> from box_embeddings.parameterizations.box_tensor import BoxTensor
        >>> z = [-2.0]*100
        >>> Z = [0.0]*100
        >>> input = [z, Z]
        >>> box1 = BoxTensor(torch.tensor(input))
        >>> print(soft_volume(box1)) # doctest: +NORMALIZE_WHITESPACE
        tensor(5.9605e+32)
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return (
        torch.prod(softplus(box_tensor.Z - box_tensor.z, beta=beta), dim=-1)
        * scale
    )


def log_soft_volume(
    box_tensor: BoxTensor, beta: float = 1.0, scale: float = 1.0
) -> torch.Tensor:
    """Volume of boxes. Uses softplus instead of ReLU/clamp

    Args:
        box_tensor: input
        beta: the beta parameter for the softplus
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1]

    Examples:
        >>> from box_embeddings.parameterizations.box_tensor import BoxTensor
        >>> z = [-2.0]*100
        >>> Z = [0.0]*100
        >>> input = [z, Z]
        >>> box1 = BoxTensor(torch.tensor(input))
        >>> log_soft_volume(box1) # doctest: +NORMALIZE_WHITESPACE
        tensor(75.4679)
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return torch.sum(
        torch.log(softplus(box_tensor.Z - box_tensor.z, beta=beta) + eps),
        dim=-1,
    ) + float(
        np.log(scale)
    )  # need this eps to that the derivative of log does not blow


@Volume.register("soft")
class SoftVolume(Volume):
    """ Softplus based volume."""

    def __init__(self, log_scale: bool = True, beta: float = 1.0) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            beta: Softplus' beta parameter.
        """
        super().__init__(log_scale)
        self.beta = beta

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        if self.log_scale:
            return log_soft_volume(box_tensor, beta=self.beta)
        else:
            return soft_volume(box_tensor, beta=self.beta)
