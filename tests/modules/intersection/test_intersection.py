from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection.intersection import Intersection
from box_embeddings.modules.intersection.hard_intersection import (
    HardIntersection,
)
from box_embeddings.modules.intersection.gumbel_intersection import (
    GumbelIntersection,
)
import torch


def test_hard_intersection() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    expected = HardIntersection()(box1, box2)
    res = Intersection()(box1, box2)
    assert res == expected


def test_gumbel_intersection() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    expected = GumbelIntersection()(box1, box2)
    res = Intersection(intersection_temperature=1.0)(box1, box2)
    assert res == expected
