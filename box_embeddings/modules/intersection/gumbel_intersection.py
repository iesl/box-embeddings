from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from torch import Tensor
from .intersection import Intersection
from box_embeddings.parameterizations import TBoxTensor
from box_embeddings.common.utils import logsumexp2
from box_embeddings import box_debug_level


def _compute_logaddexp_with_clipping(
    t1: TBoxTensor, t2: TBoxTensor, gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_z = torch.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * (lse_z)
    z = torch.max(z, torch.max(t1.z, t2.z))  # type: ignore
    lse_Z = torch.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * (lse_Z)
    Z = torch.min(Z, torch.min(t1.Z, t2.Z))

    return z, Z


def _compute_logaddexp_with_nextafter(
    t1: TBoxTensor, t2: TBoxTensor, gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Need to perform nextafter to ensure that the necessary inequlity (state below)
    # holds. We need it for conditional probability computations.
    # Note: torch.nextafter does not have gradient so we
    # will use it to produce the value but not use it during backward
    lse_z = torch.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    lse_z_value = torch.nextafter(lse_z, lse_z + 1)
    z = gumbel_beta * (lse_z_value.detach() + (lse_z - lse_z.detach()))
    lse_Z = torch.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    lse_Z_value = torch.nextafter(lse_Z, lse_Z + 1)
    Z = -gumbel_beta * (lse_Z_value.detach() + (lse_Z - lse_Z.detach()))

    return z, Z


def _compute_logaddexp(
    t1: TBoxTensor, t2: TBoxTensor, gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_z = torch.logaddexp(t1.z / gumbel_beta, t2.z / gumbel_beta)
    z = gumbel_beta * lse_z
    lse_Z = torch.logaddexp(-t1.Z / gumbel_beta, -t2.Z / gumbel_beta)
    Z = -gumbel_beta * lse_Z

    return z, Z


def gumbel_intersection(
    left: TBoxTensor,
    right: TBoxTensor,
    gumbel_beta: float = 1.0,
    approximation_mode: Optional[str] = None,
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
        approximation_mode: Use hard clipping ('clipping') or nextafter trick ('nextafter')
            to satisfy the required inequalities. (default: None)

    Returns:
         The resulting BoxTensor obtained by interection.
         It has the same concrete type as the `self` (left operand).

    Raises:
        ValueError: When approximation_mode is not in [None, 'clipping', 'nextafter']
    """
    t1 = left
    t2 = right

    if approximation_mode is None:
        z, Z = _compute_logaddexp(t1, t2, gumbel_beta)
    elif approximation_mode == "clipping":
        z, Z = _compute_logaddexp_with_clipping(t1, t2, gumbel_beta)
    elif approximation_mode == "nextafter":
        z, Z = _compute_logaddexp_with_nextafter(t1, t2, gumbel_beta)
    else:
        raise ValueError(
            f"{approximation_mode} is not a valid approximation_mode."
        )

    if box_debug_level > 0:
        with torch.no_grad():  # type:ignore
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

    def __init__(
        self, beta: float = 1.0, approximation_mode: Optional[str] = None
    ) -> None:
        """

        Args:
            beta: Gumbel's beta parameter
            approximation_mode: Use hard clipping ('clipping') or nextafter trick ('nextafter')
                to satisfy the required inequalities. (default: None)

        note: Both the approximation_modes can produce inaccurate gradients in extreme cases. Hence,
            if you are not using the result of the intersection box to compute a conditional probability
            value, it is recommended to leave the `approximation_mode` as `None`.

        """
        super().__init__()  # type: ignore
        self.beta = beta
        self.approximation_mode = approximation_mode

    def _forward(self, left: TBoxTensor, right: TBoxTensor) -> TBoxTensor:
        """Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        """

        return gumbel_intersection(
            left,
            right,
            gumbel_beta=self.beta,
            approximation_mode=self.approximation_mode,
        )
