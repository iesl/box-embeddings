from typing import List, Tuple, Union, Dict, Any, Optional

from box_embeddings.common import constant
from box_embeddings.common.registrable import Registrable
import torch

from box_embeddings.modules.volume._volume import _Volume
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.utils import tiny_value_of_dtype
from torch.nn.functional import softplus
import numpy as np

eps = tiny_value_of_dtype(torch.float)
euler_gamma = constant.EULER_GAMMA


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


@_Volume.register("bessel-approx")
class BesselApproxVolume(_Volume):
    """Uses the Softplus as an approximation of
    Bessel function.
    """

    def __init__(
        self,
        log_scale: bool = True,
        volume_temperature: float = 1.0,
        intersection_temperature: float = 1.0,
    ) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            volume_temperature: 1/volume_temperature is the beta parameter for the softplus
            intersection_temperature: the intersection_temperature parameter (same value used in intersection).
        """
        super().__init__(log_scale)
        self.volume_temperature = volume_temperature
        self.intersection_temperature = intersection_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """
        return bessel_volume_approx(
            box_tensor,
            volume_temperature=self.volume_temperature,
            intersection_temperature=self.intersection_temperature,
            log_scale=self.log_scale,
        )
