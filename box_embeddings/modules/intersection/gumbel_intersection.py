from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from torch import Tensor
from .intersection import Intersection
from box_embeddings.parameterizations import TBoxTensor


def gumbel_intersection(
    left: TBoxTensor, right: TBoxTensor, gumbel_beta: float = 1.0
) -> TBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand
        gumbel_beta: Beta parameter

    Returns:
         The resulting BoxTensor obtained by interection.
         It has the same concrete type as the `self` (left operand).
    """
    t1 = left
    t2 = right
    z = torch.max(t1.z, t2.z)
    Z = torch.min(t1.Z, t2.Z)

    z = gumbel_beta * torch.logsumexp(
        torch.stack((t1.z / gumbel_beta, t2.z / gumbel_beta)), 0
    )
    z = torch.max(z, torch.max(t1.z, t2.z))
    Z = -gumbel_beta * torch.logsumexp(
        torch.stack((-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)), 0
    )
    Z = torch.min(Z, torch.min(t1.Z, t2.Z))

    return left.from_zZ(z, Z)


@Intersection.register("gumbel")
class GumbelIntersection(Intersection):
    """Gumbel intersection operation as a Layer/Module"""

    def __init__(self, beta: float = 1.0) -> None:
        """

        Args:
            beta: Gumbel's beta parameter

        """
        self.beta = beta

    def forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """ Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        """

        return gumbel_intersection(left, right, gumbel_beta=self.beta)
