import torch

from box_embeddings.modules.volume._volume import _Volume
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.utils import tiny_value_of_dtype
from torch.nn.functional import softplus
import numpy as np

eps = tiny_value_of_dtype(torch.float)


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


@_Volume.register("soft")
class SoftVolume(_Volume):
    """ Softplus based volume."""

    def __init__(
        self, log_scale: bool = True, volume_temperature: float = 1.0
    ) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            volume_temperature: 1/volume_temperature is the beta parameter for the softplus
        """
        super().__init__(log_scale)
        self.volume_temperature = volume_temperature

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        return soft_volume(
            box_tensor,
            volume_temperature=self.volume_temperature,
            log_scale=self.log_scale,
        )
