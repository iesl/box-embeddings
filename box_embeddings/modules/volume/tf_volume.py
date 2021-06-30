from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import tensorflow as tf
from box_embeddings.common.tf_utils import tiny_value_of_dtype

from box_embeddings.modules.volume._tf_volume import _TFVolume
from box_embeddings.modules.volume.tf_bessel_volume import (
    tf_bessel_volume_approx,
)
from box_embeddings.modules.volume.tf_hard_volume import tf_hard_volume
from box_embeddings.modules.volume.tf_soft_volume import tf_soft_volume
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor


eps = tiny_value_of_dtype(tf.float64)


class TFVolume(_TFVolume):
    """One for All volume class"""

    def __init__(
        self,
        log_scale: bool = True,
        volume_temperature: float = 0.0,
        intersection_temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            volume_temperature: if non-zero, uses softplus instead of ReLU/clamp
            intersection_temperature: if non-zero, uses softplus as approximation of Bessel function
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.log_scale = log_scale
        self.volume_temperature = volume_temperature
        self.intersection_temperature = intersection_temperature

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Returns:
            torch.Tensor
        """
        if self.volume_temperature == 0 and self.intersection_temperature == 0:
            return tf_hard_volume(box_tensor, self.log_scale)
        elif self.intersection_temperature == 0:
            return tf_soft_volume(
                box_tensor, self.volume_temperature, self.log_scale
            )
        else:
            return tf_bessel_volume_approx(
                box_tensor,
                self.volume_temperature,
                self.intersection_temperature,
                self.log_scale,
            )
