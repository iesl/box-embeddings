import tensorflow as tf
from box_embeddings.modules.volume._tf_volume import _TFVolume
from box_embeddings.parameterizations import TFBoxTensor

eps = 1e-23


def tf_hard_volume(
    box_tensor: TFBoxTensor, log_scale: bool = True
) -> tf.Tensor:

    if log_scale:
        res = tf.math.reduce_sum(
            tf.math.log(
                tf.clip_by_value(
                    box_tensor.Z - box_tensor.z,
                    clip_value_min=eps,
                    clip_value_max=float('inf'),
                ),
            ),
            axis=-1,
        )

        return res
    else:

        res = tf.math.reduce_prod(
            tf.clip_by_value(
                box_tensor.Z - box_tensor.z,
                clip_value_min=eps,
                clip_value_max=float('inf'),
            ),
            axis=-1,
        )

        return res


@_TFVolume.register("hard")
class TFHardVolume(_TFVolume):

    """Hard ReLU based volume."""

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Hard ReLU base volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor

        """

        return tf_hard_volume(box_tensor, self.log_scale)
