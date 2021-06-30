from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from .tf_pooling import TFBoxPooler
import tensorflow as tf


def tf_bag_of_boxes_pooler(
    boxes: TFBoxTensor,
    mask: Optional[tf.Tensor] = None,
    weights: Optional[tf.Tensor] = None,
    dim: int = 0,
    keepdim: bool = False,
) -> TFBoxTensor:
    box_z = boxes.z
    box_Z = boxes.Z

    if weights is None:
        weights = tf.ones_like(box_z)

    if mask is not None:
        weights = weights * mask
    denominator = tf.math.reduce_sum(weights, axis=dim, keepdims=keepdim)
    z = tf.math.reduce_sum(box_z * weights, axis=dim, keepdims=keepdim) / (
        denominator + 1e-14
    )
    Z = tf.math.reduce_sum(box_Z * weights, axis=dim, keepdims=keepdim) / (
        denominator + 1e-14
    )
    return boxes.like_this_from_zZ(z, Z)


@TFBoxPooler.register("bag-of-boxes-pooler")
class TFBagOfBoxesBoxPooler(TFBoxPooler):

    """Pools a box tensor using hard intersection operation"""

    def __init__(self, dim: int = 0, keepdim: bool = False):
        super().__init__()  # type:ignore
        self.dim = dim
        self.keepdim = keepdim

    def __call__(  # type:ignore
        self,
        box_tensor: TFBoxTensor,
        mask: tf.Tensor = None,
        weights: tf.Tensor = None,
    ) -> TFBoxTensor:  # type:ignore
        """

        Args:
            box_tensor: Input
            mask: With shape as box_tensor.box_shape[:-1].
                0 at a position means mask it.
            weights:  With shape as box_tensor.box_shape[:-1].

        Returns:
            TFBoxTensor: Pooled output
        """

        return tf_bag_of_boxes_pooler(
            box_tensor,
            mask=mask,
            weights=weights,
            dim=self.dim,
            keepdim=self.keepdim,
        )
