from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.common.utils import tiny_value_of_dtype
from torch.nn.functional import softplus
import numpy as np

eps = tiny_value_of_dtype(torch.float)
euler_gamma = 0.57721566490153286060

def bessel_volume_approx(
    box_tensor: BoxTensor, beta: float = 1.0, gumbel_beta: float = 1.0,
    scale: float = 1.0
) -> torch.Tensor:
    """ Volume of boxes. Uses the Softplus as an approximation of 
        Bessel funtion.

        Args:
            box_tensor: input
            beta: the beta parameter for the softplus.
            gumbel_beta: the gumbel_beta parameter (same value used in intersection).
            scale: scale parameter. Should be left as 1.0 (default)
                in most cases.

        Returns:
            Tensor of shape (..., ) when self has shape (..., 2, num_dims)

        Raises:
            ValueError: if scale not in (0,1]
        """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return (
        torch.prod(softplus(box_tensor.Z - box_tensor.z - 2 * euler_gamma * gumbel_beta, beta=beta), dim=-1)
        * scale
    )


def log_bessel_volume_approx(
    box_tensor: BoxTensor, beta: float = 1.0, gumbel_beta: float = 1.0,
    scale: float = 1.0
) -> torch.Tensor:
    """ Volume of boxes. Uses the Softplus as an approximation of 
        Bessel funtion.

        Args:
            box_tensor: input.
            beta: the beta parameter for the softplus.
            gumbel_beta: the gumbel_beta parameter (same value used in intersection).
            scale: scale parameter. Should be left as 1.0 (default)
                in most cases.

        Returns:
            Tensor of shape (..., ) when self has shape (..., 2, num_dims)

        Raises:
            ValueError: if scale not in (0,1]
        """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return torch.sum(
        torch.log(
            softplus(box_tensor.Z - box_tensor.z - 2 * euler_gamma * gumbel_beta, beta=beta).clamp_min(eps)
        ),
        dim=-1,
    ) + float(
        np.log(scale)
    )  # need this eps to that the derivative of log does not blow


@Volume.register("bessel-approx")
class BesselApproxVolume(Volume):
    """ Softplus based volume."""

    def __init__(self, log_scale: bool = True, beta: float = 1.0, gumbel_beta: float = 1.0) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            beta: Softplus' beta parameter.
            gumbel_beta: the gumbel_beta parameter (same value used in intersection).
        """
        super().__init__(log_scale)
        self.beta = beta
        self.gumbel_beta = gumbel_beta

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """ Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        if self.log_scale:
            return log_bessel_volume_approx(box_tensor, beta=self.beta, gumbel_beta=self.gumbel_beta)
        else:
            return bessel_volume_approx(box_tensor, beta=self.beta, gumbel_beta=self.gumbel_beta)
