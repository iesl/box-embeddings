from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import TFBoxTensor
import tensorflow as tf

from box_embeddings.common.registrable import Registrable


class TFIntersection(tf.Module, Registrable):
    """Base class for intersection Layer"""

    def forward(self, left: TFBoxTensor, right: TFBoxTensor) -> TFBoxTensor:
        # broadcast if necessary
        # let the = case also be processed

        if len(left.box_shape) >= len(right.box_shape):
            right.broadcast(left.box_shape)
        else:
            left.broadcast(right.box_shape)

        return self._forward(left, right)

    def _forward(self, left: TFBoxTensor, right: TFBoxTensor) -> TFBoxTensor:
        raise NotImplementedError
