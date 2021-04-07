from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.common.tf_utils import tiny_value_of_dtype

eps = tiny_value_of_dtype(tf.float64)


class TFVolume(tf.Module, Registrable):

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

    def forward(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Raises:
            NotImplementedError: base class
        """
        raise NotImplementedError


def tf_hard_volume(box_tensor: TFBoxTensor) -> tf.Tensor:
    """Volume of boxes. Returns 0 where boxes are flipped.

    Args:
        box_tensor: input

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)
    """

    return tf.math.reduce_prod(
        tf.clip_by_value(
            box_tensor.Z - box_tensor.z,
            clip_value_min=0,
            clip_value_max=float('inf'),
        ),
        axis=-1,
    )


def tf_log_hard_volume(box_tensor: TFBoxTensor) -> tf.Tensor:
    res = tf.math.sum(
        tf.math.log(
            tf.clip_by_value(
                box_tensor.Z - box_tensor.z,
                clip_value_min=eps,
                clip_value_max=float('inf'),
            ),
            dim=-1,
        ),
        axis=1,
    )

    return res


@TFVolume.register("hard")
class TFHardVolume(TFVolume):

    """Hard ReLU based volume."""

    def forward(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Hard ReLU base volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor

        """

        if self.log_scale:
            return tf_log_hard_volume(box_tensor)
        else:
            return tf_hard_volume(box_tensor)
