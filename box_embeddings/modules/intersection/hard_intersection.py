from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from torch import Tensor
from box_embeddings.modules.intersection._intersection import _Intersection
from box_embeddings.parameterizations import TBoxTensor


def hard_intersection(left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand

    Returns:
         The resulting BoxTensor obtained by interection.
         It has the same concrete type as the `self` (left operand).
    """
    t1 = left
    t2 = right
    z = torch.max(t1.z, t2.z)
    Z = torch.min(t1.Z, t2.Z)

    return left.from_zZ(z, Z)


@_Intersection.register("hard")
class HardIntersection(_Intersection):
    """Hard intersection operation as a Layer/Module"""

    def _forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        Note:
            This function can give fipped boxes, i.e. where z[i] > Z[i]
        """

        return hard_intersection(left, right)
