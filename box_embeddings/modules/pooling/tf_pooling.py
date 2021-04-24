from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import TFBoxTensor
import tensorflow as tf

from box_embeddings.common.registrable import Registrable


class TFBoxPooler(tf.Module, Registrable):
    """Base class for Box Pooling"""

    def __call__(self, box_tensor: TFBoxTensor, **kwargs: Any) -> TFBoxTensor:
        raise NotImplementedError
