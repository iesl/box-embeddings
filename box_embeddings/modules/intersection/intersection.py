from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import BoxTensor
import torch

from box_embeddings.common.registrable import Registrable


class Intersection(torch.nn.Module, Registrable):
    """Base class for intersection Layer"""

    def forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        # broadcast if necessary
        # let the = case also be processed

        if len(left.box_shape) >= len(right.box_shape):
            right.broadcast(left.box_shape)
        else:
            left.broadcast(right.box_shape)

        return self._forward(left, right)

    def _forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        raise NotImplementedError
