from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.regularization.tf_regularizer import (
    TFBoxRegularizer,
)

eps = 1e-23


def tf_l2_side_regularizer(
    box_tensor: TFBoxTensor, log_scale: bool = False
) -> Union[float, tf.Tensor]:
    """Applies l2 regularization on all sides of all boxes and returns the sum.

    Args:
        box_tensor: TODO
        log_scale: mean in log scale

    Returns:
        (None)
    """
    z = box_tensor.z  # (..., box_dim)
    Z = box_tensor.Z  # (..., box_dim)

    if not log_scale:
        return tf.math.reduce_mean((Z - z) ** 2)
    else:
        return tf.math.reduce_mean(tf.math.log(tf.math.abs(Z - z) + eps))


@TFBoxRegularizer.register("l2_side")
class TFL2SideBoxRegularizer(TFBoxRegularizer):

    """Applies l2 regularization on side lengths."""

    def __init__(self, weight: float, log_scale: bool = False) -> None:
        """TODO: to be defined.

        Args:
            weight: Weight (hyperparameter) given to this regularization in the overall loss.
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.


        """
        super().__init__(weight, log_scale=log_scale)

    def __call__(self, box_tensor: TFBoxTensor) -> Union[float, tf.Tensor]:
        """Applies l2 regularization on all sides of all boxes and returns the sum.

        Args:
            box_tensor: TODO

        Returns:
            (None)
        """

        return tf_l2_side_regularizer(box_tensor, log_scale=self.log_scale)
