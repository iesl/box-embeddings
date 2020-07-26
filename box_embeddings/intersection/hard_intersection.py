from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from torch import Tensor
from .intersection import Intersection
from box_embeddings.parameterizations import TBoxTensor


def intersection(left: TBoxTensor, right: TBoxTensor) -> Tuple[Tensor, Tensor]:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for broadcasting

    Args:
        self: BoxTensor which is the left operand
        other: BoxTensor which is the right operand

    Returns:
        intersection: The resulting BoxTensor obtained by interection.
            It has the same concrete type as the `self` (left operand).
    """
    t1 = left
    t2 = right
    z = torch.max(t1.z, t2.z)
    Z = torch.min(t1.Z, t2.Z)

    return left.from_zZ(z, Z)


class HardIntersection(Intersection):
    """Hard intersection operation as a Layer/Module"""

    def forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """ Gives intersection of self and other.

        .. note:: This function can give fipped boxes, i.e. where z[i] > Z[i]
        """

        return intersection(left, right)
