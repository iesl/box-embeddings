from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import TBoxTensor
import torch

from box_embeddings.common.registrable import Registrable


class Intersection(torch.nn.Module, Registrable):
    """Base class for intersection Layer"""

    def forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        raise NotImplementedError
