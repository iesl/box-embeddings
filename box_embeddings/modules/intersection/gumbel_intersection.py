from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from torch import Tensor
from .intersection import Intersection
from box_embeddings.parameterizations import TBoxTensor
from box_embeddings.common.utils import logsumexp2
from box_embeddings import box_debug_level


def gumbel_intersection(
    left: TBoxTensor, right: TBoxTensor, gumbel_beta: float = 1.0
) -> TBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give fipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for automatic broadcasting

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
    # Need to perform nextafter to ensure that the necessary inequlity (state below)
    # holds. We need it for conditional probability computations.
    lse_z = torch.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * torch.nextafter(lse_z, lse_z + 1)
    lse_Z = torch.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * torch.nextafter(lse_Z, lse_Z + 1)

    if box_debug_level > 0:
        with torch.no_grad():
            assert (
                torch.max(t1.z, t2.z) < z
            ), "max(a,b) < beta*log(exp(a/beta) + exp(b/beta)) not holding"
            assert (
                torch.min(t1.z, t2.z) > Z
            ), "min(a,b) > -beta*log(exp(-a/beta) + exp(-b/beta)) not holding"

    return left.from_zZ(z, Z)


@Intersection.register("gumbel")
class GumbelIntersection(Intersection):
    """Gumbel intersection operation as a Layer/Module"""

    def __init__(self, beta: float = 1.0) -> None:
        """

        Args:
            beta: Gumbel's beta parameter

        """
        super().__init__()
        self.beta = beta

    def _forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """ Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        """

        return gumbel_intersection(left, right, gumbel_beta=self.beta)
