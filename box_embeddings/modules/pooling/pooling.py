from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import BoxTensor
import torch

from box_embeddings.common.registrable import Registrable


class BoxPooler(torch.nn.Module, Registrable):
    """Base class for Box Pooling"""

    def forward(self, box_tensor: BoxTensor, **kwargs: Any) -> BoxTensor:
        raise NotImplementedError
