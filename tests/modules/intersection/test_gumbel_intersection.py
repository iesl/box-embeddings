from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection import (
    GumbelIntersection,
    gumbel_intersection,
)
import torch
from box_embeddings.modules.intersection.hard_intersection import (
    hard_intersection,
    HardIntersection,
)
import pytest


@pytest.mark.xfail
def test_intersection() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    hard_res = BoxTensor(
        torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]).float()
    )
    gumbel_intersection(box1, box2)


# def test_intersection_with_broadcasting() -> None:
#    box1 = BoxTensor(
#        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]])
#    )  # box_shape (2,2)
#    box2 = BoxTensor(torch.tensor([[2, 0], [6, 2]]))  # box_shape (2,)
#    res_hard = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]))
#    gumbel_intersection(box1, box2)
#    gumbel_intersection(box2, box1)


@pytest.mark.xfail
def test_intersection_module() -> None:
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]))
    box2 = BoxTensor(torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]))
    GumbelIntersection(beta=2.0)(box1, box2)


@pytest.mark.xfail
def test_intersection_with_broadcasting_module() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]])
    )  # box_shape (2,2)
    box2 = BoxTensor(torch.tensor([[2, 0], [6, 2]]))  # box_shape (2,)
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]))
    GumbelIntersection()(box1, box2)
    GumbelIntersection(beta=10.0)(box2, box1)
