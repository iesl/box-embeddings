"""
    Implementation of sigmoid box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxTensor,
    TFBoxFactory,
    TFTBoxTensor,
)
from box_embeddings.common.tf_utils import softplus_inverse, inv_sigmoid
import tensorflow as tf
import warnings


def tf_index_select(input_, dim, indices):
    """
    Args:
        input_(tensor): input tensor
        dim(int): dimension
        indices(list): selected indices list

    Returns:
        Tensor
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape) - 1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0] * len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


@TFBoxFactory.register_box_class("tfsigmoid")
class TFSigmoidBoxTensor(TFBoxTensor):

    """
    Sigmoid Box Tensor
    """

    def __init__(self, data: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]):
        """

        Args:
            data: The weights for the box
        """
        super().__init__(data)

    @property
    def z(self) -> tf.Tensor:
        """Lower left coordinate as Tensor

        Returns:
            Tensor: lower left corner
        """

        if self.data is not None:
            return tf.math.sigmoid(self.data[..., 0, :])
        else:
            return self._z  # type:ignore

    @property
    def Z(self) -> tf.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        if self.data is not None:
            z = self.z
            return z + tf.math.sigmoid(self.data[..., 1, :]) * (1.0 - z)  # type: ignore
        else:
            return self._Z  # type:ignore

    @classmethod
    def W(  # type:ignore
        cls: Type[TFTBoxTensor],
        z: tf.Tensor,
        Z: tf.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tf.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        eps = 1e-07
        w1 = inv_sigmoid(
            tf.clip_by_value(z, clip_value_min=eps, clip_value_max=1.0 - eps)
        )
        w2 = inv_sigmoid(
            tf.clip_by_value(
                ((Z - z) / (1.0 - z)),
                clip_value_min=eps,
                clip_value_max=1.0 - eps,
            )
        )

        return tf.stack((w1, w2), -2)

    @classmethod
    def from_vector(  # type:ignore
        cls: Type[TFTBoxTensor], vector: tf.Tensor, *args: Any, **kwargs: Any
    ) -> TFBoxTensor:
        """Creates a box for a vector. In this base implementation the vector is split
        into two pieces and these are used as box weights.

        Args:
            vector: tensor
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            A BoxTensor

        Raises:
            ValueError: if last dimension is not even
        """
        len_dim = vector.shape[-1]
        dim = -1

        if vector.shape[-1] % 2 != 0:
            raise ValueError(
                f"The last dimension of vector should be even but is {vector.shape[-1]}"
            )

        split_point = int(len_dim / 2)
        w1 = tf_index_select(vector, dim, list(range(split_point)))
        w2 = tf_index_select(vector, dim, list(range(split_point, len_dim)))

        W: tf.Tensor = tf.stack((w1, w2), -2)

        return cls(W)


TFBoxFactory.register_box_class("sigmoid_from_zZ", "from_zZ")(
    TFSigmoidBoxTensor
)
TFBoxFactory.register_box_class("sigmoid_from_vector", "from_vector")(
    TFSigmoidBoxTensor
)
