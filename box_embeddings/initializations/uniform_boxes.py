from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
import torch
from .initializer import BoxInitializer
from box_embeddings.parameterizations.box_tensor import BoxFactory, BoxTensor


def uniform_boxes(
    dimensions: int,
    num_boxes: int,
    minimum: float = 0.0,
    maximum: float = 1.0,
    delta_min: float = 0.01,
    delta_max: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Creates uniform boxes such that each box is inside the
    bounding box defined by (minimum,maximum) in each dimension.

    Args:
        dimensions: TODO
        num_boxes: TODO
        minimum: TODO
        maximum: TODO
        delta_min: TODO
        delta_max: TODO

    Returns:
        z, Z tensors


    Raises:
        ValueError: TODO
    """

    if not (delta_min > 0):
        raise ValueError(f"Delta min should be >0 but is {delta_min}")

    if not (delta_max - delta_min > 0):
        raise ValueError(
            f"Expected: delta_max {delta_max}  > delta_min {delta_min} "
        )

    if not (delta_max <= (maximum - minimum)):
        raise ValueError(
            f"Expected: delta_max {delta_max} <= (max-min) {maximum-minimum}"
        )

    if not (maximum > minimum):
        raise ValueError(f"Expected: maximum {maximum} > minimum {minimum}")
    centers = np.random.uniform(
        minimum + delta_max / 2.0 + 1e-8,
        maximum - delta_max / 2.0 - 1e-8,
        size=(num_boxes, dimensions),
    )

    deltas = np.random.uniform(
        delta_min, delta_max, size=(num_boxes, dimensions)
    )
    z = centers - deltas / 2.0
    Z = centers + deltas / 2.0
    assert (z >= minimum).all()
    assert (Z <= maximum).all()

    return torch.tensor(z), torch.tensor(Z)


class UniformBoxInitializer(BoxInitializer):

    """Docstring for UniformBoxInitializer. """

    def __init__(
        self,
        dimensions: int,
        num_boxes: int,
        box_type_factory: BoxFactory,
        minimum: float = 0.0,
        maximum: float = 1.0,
        delta_min: float = 0.01,
        delta_max: float = 0.5,
    ) -> None:
        """TODO: Docstring for __init__.

        Args:
            dimensions: TODO
            num_boxes: TODO
            box_type_factory: TODO
            minimum: TODO
            maximum: TODO
            delta_min: TODO
            delta_max: TODO

        Returns: (None)

        """
        self.dimensions = dimensions
        self.num_boxes = num_boxes
        self.minimum = minimum
        self.maximum = maximum
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.box_type_factory = box_type_factory

    def __call__(self, t: BoxTensor) -> None:  # type:ignore
        z, Z = uniform_boxes(
            self.dimensions,
            self.num_boxes,
            self.minimum,
            self.maximum,
            self.delta_min,
            self.delta_max,
        )
        with torch.no_grad():
            W = self.box_type_factory.box_subclass.W(z, Z)  # type: ignore
            t.reinit(W)
