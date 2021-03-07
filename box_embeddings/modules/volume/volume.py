from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.utils import tiny_value_of_dtype
import box_embeddings.common.constant as constant
import numpy as np
from torch.nn.functional import softplus

eps = tiny_value_of_dtype(torch.float)
euler_gamma = constant.EULER_GAMMA


class Volume(torch.nn.Module, Registrable):
    """Base volume class"""

    default_implementation = "hard"

    def __init__(
        self,
        log_scale: bool = True,
        volume_temperature: float = 0.0,
        intersection_temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            volume_temperature: if non-zero, uses softplus instead of ReLU/clamp
            intersection_temperature: if non-zero, uses softplus as approximation of Bessel function
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.log_scale = log_scale
        self.volume_temperature = volume_temperature
        self.intersection_temperature = intersection_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Returns:
            torch.Tensor
        """
        if self.volume_temperature == 0 and self.intersection_temperature == 0:
            return hard_volume(box_tensor, self.log_scale)
        elif self.intersection_temperature == 0:
            return soft_volume(
                box_tensor, self.volume_temperature, self.log_scale
            )
        else:
            return bessel_volume_approx(
                box_tensor,
                self.volume_temperature,
                self.intersection_temperature,
                self.log_scale,
            )


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


def soft_volume(
    box_tensor: BoxTensor,
    volume_temperature: float = 1.0,
    log_scale: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """Volume of boxes. Uses softplus instead of ReLU/clamp

    Args:
        box_tensor: input
        volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        log_scale: Whether the output should be in log scale or not.
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1] or volume_temperature is 0.
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    if volume_temperature == 0:
        raise ValueError("volume_temperature must be non-zero")

    if log_scale:
        return torch.sum(
            torch.log(
                softplus(
                    box_tensor.Z - box_tensor.z, beta=1 / volume_temperature
                )
                + eps
            ),
            dim=-1,
        ) + float(
            np.log(scale)
        )  # need this eps to that the derivative of log does not blow

    return (
        torch.prod(
            softplus(box_tensor.Z - box_tensor.z, beta=1 / volume_temperature),
            dim=-1,
        )
        * scale
    )


def bessel_volume_approx(
    box_tensor: BoxTensor,
    volume_temperature: float = 1.0,
    intersection_temperature: float = 1.0,
    log_scale: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """Volume of boxes. Uses the Softplus as an approximation of
    Bessel funtion.

    Args:
        box_tensor: input
        volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        intersection_temperature: the intersection_temperature parameter (same value used in intersection).
        log_scale: Whether the output should be in log scale or not.
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1] or volume_temperature is 0.
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    if volume_temperature == 0:
        raise ValueError("volume_temperature must be non-zero")

    if log_scale:
        return torch.sum(
            torch.log(
                softplus(
                    box_tensor.Z
                    - box_tensor.z
                    - 2 * euler_gamma * intersection_temperature,
                    beta=1 / volume_temperature,
                )
                + eps
            ),
            dim=-1,
        ) + float(
            np.log(scale)
        )  # need this eps to that the derivative of log does not blow

    return (
        torch.prod(
            softplus(
                box_tensor.Z
                - box_tensor.z
                - 2 * euler_gamma * intersection_temperature,
                beta=1 / volume_temperature,
            ),
            dim=-1,
        )
        * scale
    )
