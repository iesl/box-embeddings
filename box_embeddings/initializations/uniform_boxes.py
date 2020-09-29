from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
import torch


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
