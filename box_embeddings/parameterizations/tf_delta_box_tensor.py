"""
    Implementation of min-delta box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type

# from box_embeddings.common.utils import softplus_inverse
from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxTensor,
    TFBoxFactory,
    TFTBoxTensor,
)
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


def _box_shape_ok(t: tf.Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.shape[-2] != 2:
            return False

        return True


def softplus_inverse(
    t: tf.Tensor, beta: float = 1.0, threshold: float = 20
) -> tf.Tensor:
    below_thresh = beta * t < threshold
    res = t
    # res[below_thresh] = (
    # torch.log(torch.exp(beta * t[below_thresh]) - 1.0) / beta
    # )
    res[below_thresh] = (
        tf.math.log(
            tf.clip_by_value(
                tf.math.expm1(beta * t[below_thresh]),
                clip_value_min=1e-323,
                clip_value_max=float('inf'),
            )
        )
        / beta
    )

    return res


@TFBoxFactory.register_box_class("tfmindelta")
class TFMinDeltaBoxTensor(TFBoxTensor):

    """Unconstrained min-delta box tensor.

    For input of the shape (..., 2, box_dim), this parameterization
    defines z=w, and Z=w + delta, where w and delta come from the -2th dimension
    of the input. It uses softplus to keep the delta positive.

    """

    def __init__(
        self,
        data: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        beta: float = 1.0,
        threshold: float = 20,
    ):
        """

        Args:
            data: The weights for the box
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta
        """
        super().__init__(data)
        self.beta = beta
        self.threshold = threshold

    @property
    def kwargs(self) -> Dict:
        return {"beta": self.beta, "threshold": self.threshold}

    @property
    def args(self) -> Tuple:
        return tuple()

    @property
    def Z(self) -> tf.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        if self.data is not None:
            return (
                self.z
                + tf.math.softplus(self.data[..., 1, :] * self.beta)
                / self.beta
            )

            # return self.z + torch.nn.functional.softplus(
            #    self.data[..., 1, :], beta=self.beta, threshold=self.threshold
            # )
        else:
            return self._Z  # type:ignore

    @classmethod
    def W(  # type:ignore
        cls: Type[TFTBoxTensor],
        z: tf.Tensor,
        Z: tf.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> tf.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        The min coordinate is stored as is:
        W[...,0,:] = z
        W[...,1,:] = softplus_inverse(Z-z)

        The max coordinate is transformed


        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            beta: TODO
            threshold: TODO

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        if ((Z - z) < 0).any():
            warnings.warn(
                "W() method for TFMinDeltaBoxTensor is numerically unstable."
                " It can produce high error when input Z-z is < 0."
            )

        return tf.stack(
            (z, softplus_inverse(Z - z, beta=beta, threshold=threshold)), -2
        )

    #    @classmethod
    #    def from_zZ(  # type:ignore
    #        cls,
    #        z: torch.Tensor,
    #        Z: torch.Tensor,
    #        beta: float = 1.0,
    #        threshold: float = 20,
    #    ) -> BoxTensor:
    #        """Creates a box for the given min-max coordinates (z,Z).
    #
    #        In the this base implementation we do this by
    #        stacking z and Z along -2 dim to form W.
    #
    #        Args:
    #            z: lower left
    #            Z: top right
    #            beta: beta parameter for softplus for delta. Depending on the
    #                universe box and your inputs ranges, you might want to change this.
    #                Higher values of beta will make softplus harder and bring it close to ReLU.
    #            threshold: parameter for the softplus for delta
    #
    #
    #        Returns:
    #            A BoxTensor
    #
    #        """
    #
    #        return cls((z, Z), beta=beta, threshold=threshold)

    #    def like_this_from_zZ(
    #        self, z: torch.Tensor, Z: torch.Tensor,
    #    ) -> "BoxTensor":
    #        """Creates a box for the given min-max coordinates (z,Z).
    #        This is similar to the class method :method:`from_zZ`, but
    #        uses the attributes on self and not external args, kwargs.
    #
    #        Args:
    #            z: lower left
    #            Z: top right
    #
    #        Returns:
    #            A BoxTensor
    #
    #        """
    #
    #        return self.from_zZ(z, Z, beta=self.beta, threshold=self.threshold)

    @classmethod
    def from_vector(  # type:ignore
        cls, vector: tf.Tensor, beta: float = 1.0, threshold: float = 20
    ) -> TFBoxTensor:
        """Creates a box for a vector. In this base implementation the vector is split
        into two pieces and these are used as z and delta.

        Args:
            vector: tensor
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta


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

        z = tf_index_select(vector, dim, tf.constant(list(range(split_point))))

        delta = tf_index_select(
            vector, dim, tf.constant(list(range(split_point, len_dim)))
        )

        return cls(
            tf.stack((z, delta), -2), beta=beta, threshold=threshold
        )  # type:ignore


TFBoxFactory.register_box_class("mindelta_from_zZ", "from_zZ")(
    TFMinDeltaBoxTensor
)
TFBoxFactory.register_box_class("mindelta_from_vector", "from_vector")(
    TFMinDeltaBoxTensor
)
