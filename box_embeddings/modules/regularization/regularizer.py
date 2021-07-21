from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor


class BoxRegularizer(torch.nn.Module, Registrable):

    """Base box-regularizer class"""

    def __init__(
        self,
        weight: float,
        log_scale: bool = True,
        reduction: str = 'sum',
        **kwargs: Any,
    ) -> None:
        """
        Args:
            weight: Weight (hyperparameter) given to this regularization in the overall loss.
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            reduction: Specifies the reduction to apply to the output: 'mean': the sum of the output will be divided by
                the number of elements in the output, 'sum': the output will be summed. Default: 'sum'
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.weight = weight
        self.log_scale = log_scale
        self.reduction = reduction

    def forward(self, box_tensor: BoxTensor) -> Union[float, torch.Tensor]:
        """Calls the _forward and multiplies the weight

        Args:
            box_tensor: Input box tensor

        Returns:
            scalar regularization loss
        """

        return self.weight * self._reduce(self._forward(box_tensor))

    def _forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        raise NotImplementedError

    def _reduce(self, reg_unreduced: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return torch.sum(reg_unreduced)
        elif self.reduction == "mean":
            return torch.mean(reg_unreduced)
        else:
            raise ValueError
