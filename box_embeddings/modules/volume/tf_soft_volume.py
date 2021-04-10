from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.volume.tf_volume import TFVolume
from box_embeddings.common.tf_utils import tiny_value_of_dtype
import numpy as np

eps = 1e-23


def tf_soft_volume(
    box_tensor: TFBoxTensor, beta: float = 1.0, scale: float = 1.0
) -> tf.Tensor:
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
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return (
        tf.math.reduce_prod(
            tf.math.softplus((box_tensor.Z - box_tensor.z) * beta), axis=-1
        )
        * scale
    )


def tf_log_soft_volume(
    box_tensor: TFBoxTensor, beta: float = 1.0, scale: float = 1.0
) -> tf.Tensor:
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
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return tf.math.reduce_sum(
        tf.math.log(
            tf.math.softplus((box_tensor.Z - box_tensor.z) * beta) + eps
        ),
        axis=-1,
    ) + float(
        np.log(scale)
    )  # need this eps to that the derivative of log does not blow


@TFVolume.register("soft")
class TFSoftVolume(TFVolume):
    """ Softplus based volume."""

    def __init__(self, log_scale: bool = True, beta: float = 1.0) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            beta: Softplus' beta parameter.
        """
        super().__init__(log_scale)
        self.beta = beta

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        if self.log_scale:
            return tf_log_soft_volume(box_tensor, beta=self.beta)
        else:
            return tf_soft_volume(box_tensor, beta=self.beta)
