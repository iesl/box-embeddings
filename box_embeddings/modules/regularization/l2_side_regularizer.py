from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.regularization.regularizer import BoxRegularizer

eps = 1e-23


def l2_side_regularizer(
    box_tensor: BoxTensor, log_scale: bool = False
) -> Union[float, torch.Tensor]:
    """Applies l2 regularization on all sides of all boxes and returns the sum.

    Args:
        box_tensor: TODO
        log_scale: mean in log scale

    Returns:
        (None)
    """
    z = box_tensor.z  # (..., box_dim)
    Z = box_tensor.Z  # (..., box_dim)

    if not log_scale:
        return torch.mean((Z - z) ** 2)
    else:
        return torch.mean(torch.log(torch.abs(Z - z) + eps))


@BoxRegularizer.register("l2_side")
class L2SideBoxRegularizer(BoxRegularizer):

    """Applies l2 regularization on side lengths."""

    def __init__(self, weight: float, log_scale: bool = False) -> None:
        """TODO: to be defined.

        Args:
            weight: Weight (hyperparameter) given to this regularization in the overall loss.
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.


        """
        super().__init__(weight, log_scale=log_scale)

    def _forward(self, box_tensor: BoxTensor) -> Union[float, torch.Tensor]:
        """Applies l2 regularization on all sides of all boxes and returns the sum.

        Args:
            box_tensor: TODO

        Returns:
            (None)
        """

        return l2_side_regularizer(box_tensor, log_scale=self.log_scale)
