from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from .tf_pooling import TFBoxPooler
import tensorflow as tf


def tf_hard_intersection_pooler(
    boxes: TFBoxTensor,
    mask: tf.Tensor = None,
    dim: int = 0,
    keepdim: bool = False,
) -> TFBoxTensor:
    box_z = boxes.z
    box_Z = boxes.Z

    if mask is not None:
        box_z[mask] -= float("inf")
        box_Z[mask] += float("inf")
    z = tf.math.reduce_max(box_z, axis=dim, keepdims=keepdim)[0]
    Z = tf.math.reduce_min(box_Z, axis=dim, keepdims=keepdim)[0]

    return boxes.like_this_from_zZ(z, Z)


@TFBoxPooler.register("hard-intersection")
class TFHardIntersectionBoxPooler(TFBoxPooler):

    """Pools a box tensor using hard intersection operation"""

    def __init__(self, dim: int = 0, keepdim: bool = False):
        super().__init__()  # type:ignore
        self.dim = dim
        self.keepdim = keepdim

    def __call__(  # type:ignore
        self, box_tensor: TFBoxTensor, mask: tf.Tensor = None
    ) -> TFBoxTensor:
        return tf_hard_intersection_pooler(
            box_tensor, mask=mask, dim=self.dim, keepdim=self.keepdim
        )
