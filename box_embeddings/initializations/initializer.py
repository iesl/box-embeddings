from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor


class BoxInitializer(object):

    """A base class interface which will initialize a :class:`torch.Tensor` or :class:`torch.nn.Parameter`
    """

    def __call__(self, t: BoxTensor, **kwargs: Any) -> None:
        pass
