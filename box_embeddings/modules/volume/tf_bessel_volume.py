from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.volume.tf_volume import TFVolume
from box_embeddings.common.tf_utils import tiny_value_of_dtype
import numpy as np

# eps = tiny_value_of_dtype(torch.float)
eps = 1e-23
euler_gamma = 0.57721566490153286060


def tf_bessel_volume_approx(
    box_tensor: TFBoxTensor,
    beta: float = 1.0,
    gumbel_beta: float = 1.0,
    scale: float = 1.0,
) -> tf.Tensor:
    """Volume of boxes. Uses the Softplus as an approximation of
    Bessel funtion.

    Args:
        box_tensor: input
        beta: the beta parameter for the softplus.
        gumbel_beta: the gumbel_beta parameter (same value used in intersection).
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1]
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return (
        tf.math.reduce_prod(
            tf.math.softplus(
                (box_tensor.Z - box_tensor.z - 2 * euler_gamma * gumbel_beta)
                * beta
            ),
            axis=-1,
        )
        * scale
    )


def tf_log_bessel_volume_approx(
    box_tensor: TFBoxTensor,
    beta: float = 1.0,
    gumbel_beta: float = 1.0,
    scale: float = 1.0,
) -> tf.Tensor:
    """Volume of boxes. Uses the Softplus as an approximation of
    Bessel funtion.

    Args:
        box_tensor: input.
        beta: the beta parameter for the softplus.
        gumbel_beta: the gumbel_beta parameter (same value used in intersection).
        scale: scale parameter. Should be left as 1.0 (default)
            in most cases.

    Returns:
        Tensor of shape (..., ) when self has shape (..., 2, num_dims)

    Raises:
        ValueError: if scale not in (0,1]
    """

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale should be in (0,1] but is {scale}")

    return tf.math.reduce_sum(
        tf.math.log(
            tf.math.softplus(
                (box_tensor.Z - box_tensor.z - 2 * euler_gamma * gumbel_beta)
                * beta
            )
            + eps
        ),
        axis=-1,
    ) + float(
        np.log(scale)
    )  # need this eps to that the derivative of log does not blow


@TFVolume.register("bessel-approx")
class TFBesselApproxVolume(TFVolume):
    """Uses the Softplus as an approximation of
    Bessel function.
    """

    def __init__(
        self,
        log_scale: bool = True,
        beta: float = 1.0,
        gumbel_beta: float = 1.0,
    ) -> None:
        """

        Args:
            log_scale: Where the output should be in log scale.
            beta: Softplus' beta parameter.
            gumbel_beta: the gumbel_beta parameter (same value used in intersection).
        """
        super().__init__(log_scale)
        self.beta = beta
        self.gumbel_beta = gumbel_beta

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Soft softplus base (instead of ReLU) volume.

        Args:
            box_tensor: TODO

        Returns:
            torch.Tensor
        """

        if self.log_scale:
            return tf_log_bessel_volume_approx(
                box_tensor, beta=self.beta, gumbel_beta=self.gumbel_beta
            )
        else:
            return tf_bessel_volume_approx(
                box_tensor, beta=self.beta, gumbel_beta=self.gumbel_beta
            )
