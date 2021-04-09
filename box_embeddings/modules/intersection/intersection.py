from typing import List, Tuple, Union, Dict, Any, Optional

from box_embeddings.modules.intersection._intersection import _Intersection
from box_embeddings.modules.intersection.gumbel_intersection import (
    gumbel_intersection,
)
from box_embeddings.modules.intersection.hard_intersection import (
    hard_intersection,
)
from box_embeddings.parameterizations import BoxTensor, TBoxTensor
import torch
from box_embeddings import box_debug_level
from box_embeddings.common.registrable import Registrable


@_Intersection.register("intersection")
class Intersection(_Intersection):
    """All for one Intersection operation as Layer/Module"""

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
            approximation_mode: Only for gumbel intersection,
                i.e. if intersection_temperature is non-zero Use hard clipping ('clipping') or hard clipping with
                separate value for forward and backward passes  ('clipping_forward') to satisfy the required
                inequalities. Set `None` to not use any approximation. (default: `None`)

        Note:
            Both the approximation_modes can produce inaccurate gradients in extreme cases. Hence,
            if you are not using the result of the intersection box to compute a conditional probability
            value, it is recommended to leave the `approximation_mode` as `None`.
        """
        super().__init__()  # type: ignore
        self.intersection_temperature = intersection_temperature
        self.approximation_mode = approximation_mode

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
