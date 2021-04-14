from typing import List, Tuple, Union, Dict, Any, Optional
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor


class TFBoxInitializer(object):

    """A base class interface which will initialize a :class:`torch.Tensor` or :class:`torch.nn.Parameter`"""

    def __call__(self, t: tf.Tensor, **kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with attributes {self.__dict__}"
