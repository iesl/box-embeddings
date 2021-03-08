from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.parameterizations import BoxTensor, TBoxTensor
import torch
from box_embeddings import box_debug_level
from box_embeddings.common.registrable import Registrable


class Intersection(torch.nn.Module, Registrable):
    """Intersection operation as Layer/Module"""

    def __init__(
        self,
        intersection_temperature: float = 0.0,
        approximation_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            intersection_temperature: Gumbel's beta parameter, if non-zero performs the intersection operation as
            described in `Improving Local Identifiability in Probabilistic Box Embeddings
                <https://arxiv.org/abs/2010.04831>`_
            approximation_mode: Only for gumbel intersection, i.e. if intersection_temperature is non-zero
                Use hard clipping ('clipping') or hard clipping with separate value
                for forward and backward passes  ('clipping_forward') to satisfy the required inequalities.
                Set `None` to not use any approximation. (default: `None`)

        note:
            Both the approximation_modes can produce inaccurate gradients in extreme cases. Hence,
            if you are not using the result of the intersection box to compute a conditional probability
            value, it is recommended to leave the `approximation_mode` as `None`.

        """
        super().__init__()  # type: ignore
        self.intersection_temperature = intersection_temperature
        self.approximation_mode = approximation_mode

    def forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        # broadcast if necessary
        # let the = case also be processed

        if len(left.box_shape) >= len(right.box_shape):
            right.broadcast(left.box_shape)
        else:
            left.broadcast(right.box_shape)

        return self._forward(left, right)

    def _forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        """Gives intersection of self and other.

        Args:
            left: First operand for intersection
            right: Second operand

        Returns:
            Intersection box

        Note:
            This function can give flipped boxes, i.e. where z[i] > Z[i]
        """
        if self.intersection_temperature == 0:
            return hard_intersection(left, right)
        else:
            return gumbel_intersection(
                left,
                right,
                self.intersection_temperature,
                self.approximation_mode,
            )


def hard_intersection(left: TBoxTensor, right: TBoxTensor) -> BoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give flipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand

    Returns:
         The resulting BoxTensor obtained by intersection.
    """
    t1 = left
    t2 = right
    z = torch.max(t1.z, t2.z)
    Z = torch.min(t1.Z, t2.Z)

    return BoxTensor.from_zZ(z, Z)


def gumbel_intersection(
    left: TBoxTensor,
    right: TBoxTensor,
    intersection_temperature: float = 1.0,
    approximation_mode: Optional[str] = None,
) -> TBoxTensor:
    """Hard Intersection operation as a function.

    note:
        This function can give flipped boxes, i.e. where z[i] > Z[i]

    todo:
        Add support for automatic broadcasting

    Args:
        left: BoxTensor which is the left operand
        right: BoxTensor which is the right operand
        intersection_temperature: gumbel's beta parameter
        approximation_mode: Use hard clipping ('clipping') or hard clipping with separate value
            for forward and backward passes  ('clipping_forward') to satisfy the required inequalities.
            Set `None` to not use any approximation. (default: `None`)

    Returns:
         The resulting BoxTensor obtained by interaction.
         It has the same concrete type as the `self` (left operand).

    Raises:
        ValueError: When intersection_temperature is 0 or
            approximation_mode is not in [None, 'clipping', 'clipping_forward']
    """
    t1 = left
    t2 = right

    if intersection_temperature == 0:
        raise ValueError("intersection_temperature must be non-zero.")

    if approximation_mode is None:
        z, Z = _compute_logaddexp(t1, t2, intersection_temperature)
    elif approximation_mode == "clipping":
        z, Z = _compute_logaddexp_with_clipping(
            t1, t2, intersection_temperature
        )
    elif approximation_mode == "clipping_forward":
        z, Z = _compute_logaddexp_with_clipping_and_separate_forward(
            t1, t2, intersection_temperature
        )
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


def _compute_logaddexp_with_clipping_and_separate_forward(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.z, -t1.Z), -2)
    t2_data = torch.stack((t2.z, -t2.Z), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    z = intersection_temperature * lse[..., 0, :]
    Z = -intersection_temperature * lse[..., 1, :]

    z_value = torch.max(z, torch.max(t1.z, t2.z))  # type: ignore
    Z_value = torch.min(Z, torch.min(t1.Z, t2.Z))

    z_final = (z - z.detach()) + z_value.detach()
    Z_final = (Z - Z.detach()) + Z_value.detach()

    return z_final, Z_final


def _compute_logaddexp_with_clipping(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.z, -t1.Z), -2)
    t2_data = torch.stack((t2.z, -t2.Z), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    z = intersection_temperature * lse[..., 0, :]
    Z = -intersection_temperature * lse[..., 1, :]

    z_value = torch.max(z, torch.max(t1.z, t2.z))  # type: ignore
    Z_value = torch.min(Z, torch.min(t1.Z, t2.Z))

    return z_value, Z_value


def _compute_logaddexp(
    t1: TBoxTensor, t2: TBoxTensor, intersection_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((t1.z, -t1.Z), -2)
    t2_data = torch.stack((t2.z, -t2.Z), -2)
    lse = torch.logaddexp(
        t1_data / intersection_temperature, t2_data / intersection_temperature
    )

    z = intersection_temperature * lse[..., 0, :]
    Z = -intersection_temperature * lse[..., 1, :]

    return z, Z
