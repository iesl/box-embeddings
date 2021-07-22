from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor

from box_embeddings.modules.volume._tf_volume import _TFVolume
import numpy as np

eps = 1e-23


def tf_soft_volume(
    box_tensor: TFBoxTensor,
    volume_temperature: float = 1.0,
    log_scale: bool = True,
    scale: float = 1.0,
) -> tf.Tensor:
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
        return tf.math.reduce_sum(
            tf.math.log(
                tf.math.softplus(
                    (box_tensor.Z - box_tensor.z) * (1 / volume_temperature)
                )
                + eps
            ),
            axis=-1,
        ) + float(
            np.log(scale)
        )  # need this eps to that the derivative of log does not blow

    return (
        tf.math.reduce_prod(
            tf.math.softplus(
                (box_tensor.Z - box_tensor.z) * (1 / volume_temperature)
            ),
            axis=-1,
        )
        * scale
    )

    return (
        tf.math.reduce_prod(
            tf.math.softplus(
                (box_tensor.Z - box_tensor.z) * (1 / volume_temperature)
            ),
            axis=-1,
        )
        * scale
    )


@_TFVolume.register("soft")
class TFSoftVolume(_TFVolume):
    """Softplus based volume."""

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

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        return tf_soft_volume(
            box_tensor,
            volume_temperature=self.volume_temperature,
            log_scale=self.log_scale,
        )
