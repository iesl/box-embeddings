from typing import List, Tuple, Union, Dict, Any, Optional
import tensorflow as tf
from box_embeddings.modules.intersection._tf_intersection import (
    _TFIntersection,
)
from box_embeddings.parameterizations import TFTBoxTensor
from box_embeddings.common.tf_utils import logsumexp2
from box_embeddings import box_debug_level
import numpy as np


def _tf_compute_logaddexp_with_clipping_and_separate_forward(
    t1: TFTBoxTensor, t2: TFTBoxTensor, gumbel_beta: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    lse_z = np.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * (lse_z)
    z_value = tf.math.maximum(z, tf.math.maximum(t1.z, t2.z))  # type: ignore
    z_final = (z - tf.stop_gradient(z)) + tf.stop_gradient(z_value)
    lse_Z = np.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * (lse_Z)
    Z_value = tf.math.minimum(Z, tf.math.minimum(t1.Z, t2.Z))
    Z_final = (Z - tf.stop_gradient(Z)) + tf.stop_gradient(Z_value)

    return z_final, Z_final


def _tf_compute_logaddexp_with_clipping(
    t1: TFTBoxTensor, t2: TFTBoxTensor, gumbel_beta: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    lse_z = np.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * (lse_z)
    z_value = tf.math.maximum(z, tf.math.maximum(t1.z, t2.z))  # type: ignore
    lse_Z = np.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * (lse_Z)
    Z_value = tf.math.minimum(Z, tf.math.minimum(t1.Z, t2.Z))
    return z_value, Z_value


def _tf_compute_logaddexp(
    t1: TFTBoxTensor, t2: TFTBoxTensor, gumbel_beta: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    lse_z = np.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * lse_z
    lse_Z = np.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * lse_Z

    return z, Z


def tf_gumbel_intersection(
    left: TFTBoxTensor,
    right: TFTBoxTensor,
    gumbel_beta: float = 1.0,
    approximation_mode: Optional[str] = None,
) -> TFTBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for automatic broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand
        gumbel_beta: Beta parameter
        approximation_mode: Use hard clipping ('clipping') or hard clipping with separeate value
            for forward and backward passes  ('clipping_forward') to satisfy the required inequalities.
            Set `None` to not use any approximation. (default: `None`)

    Returns:
         The resulting BoxTensor obtained by interection.
         It has the same concrete type as the `self` (left operand).

    Raises:
        ValueError: When approximation_mode is not in [None, 'clipping', 'clipping_forward']
    """
    t1 = left
    t2 = right

    if approximation_mode is None:
        z, Z = _tf_compute_logaddexp(t1, t2, gumbel_beta)
    elif approximation_mode == "clipping":
        z, Z = _tf_compute_logaddexp_with_clipping(t1, t2, gumbel_beta)
    elif approximation_mode == "clipping_forward":
        z, Z = _tf_compute_logaddexp_with_clipping_and_separate_forward(
            t1, t2, gumbel_beta
        )
    else:
        raise ValueError(
            f"{approximation_mode} is not a valid approximation_mode."
        )

    if box_debug_level > 0:
        assert (
            tf.math.maximum(t1.z, t2.z) < z
        ), "max(a,b) < beta*log(exp(a/beta) + exp(b/beta)) not holding"
        assert (
            tf.math.minimum(t1.z, t2.z) > Z
        ), "min(a,b) > -beta*log(exp(-a/beta) + exp(-b/beta)) not holding"

    return left.from_zZ(z, Z)


@_TFIntersection.register("gumbel")
class TFGumbelIntersection(_TFIntersection):
    """Gumbel intersection operation as a Layer/Module.

    Performs the intersection operation as described in `Improving Local Identifiability in Probabilistic Box Embeddings <https://arxiv.org/abs/2010.04831>`_ .
    """

    def __init__(
        self, beta: float = 1.0, approximation_mode: Optional[str] = None
    ) -> None:
        """
        Args:
            beta: Gumbel's beta parameter
            approximation_mode: Use hard clipping ('clipping') or hard clipping with separeate value
                for forward and backward passes  ('clipping_forward') to satisfy the required inequalities.
                Set `None` to not use any approximation. (default: `None`)

        note:
            Both the approximation_modes can produce inaccurate gradients in extreme cases. Hence,
            if you are not using the result of the intersection box to compute a conditional probability
            value, it is recommended to leave the `approximation_mode` as `None`.

        """
        super().__init__()  # type: ignore
        self.beta = beta
        self.approximation_mode = approximation_mode

    def __call__(
        self, left: TFTBoxTensor, right: TFTBoxTensor
    ) -> TFTBoxTensor:
        """Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        """

        return tf_gumbel_intersection(
            left,
            right,
            gumbel_beta=self.beta,
            approximation_mode=self.approximation_mode,
        )
