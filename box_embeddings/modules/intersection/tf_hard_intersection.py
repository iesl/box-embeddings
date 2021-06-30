from typing import List, Tuple, Union, Dict, Any, Optional
import tensorflow as tf
from box_embeddings.parameterizations import TFTBoxTensor
from box_embeddings.modules.intersection._tf_intersection import (
    _TFIntersection,
)


def tf_hard_intersection(
    left: TFTBoxTensor, right: TFTBoxTensor
) -> TFTBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand

    Returns:
         The resulting BoxTensor obtained by interection.
         It has the same concrete type as the `self` (left operand).
    """
    t1 = left
    t2 = right
    z = tf.math.maximum(t1.z, t2.z)
    Z = tf.math.minimum(t1.Z, t2.Z)

    return left.from_zZ(z, Z)


@_TFIntersection.register("hard")
class TFHardIntersection(_TFIntersection):
    """Hard intersection operation as a Layer/Module"""

    def __call__(
        self, left: TFTBoxTensor, right: TFTBoxTensor
    ) -> TFTBoxTensor:
        """Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        Note:
            This function can give fipped boxes, i.e. where z[i] > Z[i]
        """

        return tf_hard_intersection(left, right)
