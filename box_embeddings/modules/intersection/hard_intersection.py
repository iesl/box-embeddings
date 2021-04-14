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

    Example:
        >>> import torch
        >>> from box_embeddings.parameterizations.box_tensor import BoxTensor
        >>> box1_z = [-2.0]*10
        >>> box1_Z = [0.0]*10
        >>> data_x = torch.tensor([box1_z, box1_Z])
        >>> box1 = BoxTensor(torch.tensor([box1_z, box1_Z]))
        >>> box2_z = [1/n for n in range(1, 11)]
        >>> box2_Z = [1 - k for k in reversed(box2_z)]
        >>> box2 = BoxTensor(torch.tensor([box2_z, box2_Z]))
        >>> print(hard_intersection(box1, box2)) # doctest: +NORMALIZE_WHITESPACE
        BoxTensor(z=tensor([1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250, 0.1111,
            0.1000]),
            Z=tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
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
