from typing import Any

from box_embeddings.parameterizations import TFBoxTensor
import tensorflow as tf

from box_embeddings.common.registrable import Registrable


class _TFVolume(tf.Module, Registrable):
    """Base class for Volume Layer"""

    def __init__(self, log_scale: bool = True, **kwargs: Any) -> None:
        """
        Args:
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.log_scale = log_scale

    def __call__(self, box_tensor: TFBoxTensor) -> tf.Tensor:
        """Base implementation is hard (ReLU) volume.

        Args:
            box_tensor: Input box tensor

        Raises:
            NotImplementedError: base class
        """
        raise NotImplementedError
