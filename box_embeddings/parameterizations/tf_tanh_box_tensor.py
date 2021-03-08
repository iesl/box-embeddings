"""
    Implementation of Tanh box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxTensor,
    TFBoxFactory,
    TFTBoxTensor,
)
import box_embeddings.common.constant as constant
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


@TFBoxFactory.register_box_class("tftanh")
class TFTanhBoxTensor(TFBoxTensor):

    """
    Tanh Activated Box Tensor
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
            return (self.data[..., 0, :] + 1) / 2
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
            return z + (self.data[..., 1, :] + 1) * (1.0 - z) / 2  # type: ignore
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

        tanh_eps = constant.TANH_EPS
        z_ = tf.clip_by_value(
            z, clip_value_min=0.0, clip_value_max=1.0 - tanh_eps / 2.0
        )
        Z_ = tf.clip_by_value(
            Z, clip_value_min=tanh_eps / 2.0, clip_value_max=1.0
        )
        w1 = 2 * z_ - 1
        w2 = 2 * (Z_ - z_) / (1.0 - z_) - 1

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

        tanh_eps = constant.TANH_EPS
        split_point = int(len_dim / 2)

        w1 = tf.clip_by_value(
            tf_index_select(vector, dim, list(range(split_point))),
            clip_value_min=-1,
            clip_value_max=1.0 - tanh_eps,
        )
        w2 = tf.clip_by_value(
            tf_index_select(vector, dim, list(range(split_point, len_dim))),
            clip_value_min=-1.0 + tanh_eps,
            clip_value_max=1.0,
        )

        W: tf.Tensor = tf.stack((w1, w2), -2)

        return cls(W)


TFBoxFactory.register_box_class("tanh_from_zZ", "from_zZ")(TFTanhBoxTensor)
TFBoxFactory.register_box_class("tanh_from_vector", "from_vector")(
    TFTanhBoxTensor
)
