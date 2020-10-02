from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection.hard_intersection import (
    hard_intersection,
    HardIntersection,
)
import torch


def test_intersection() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]).float())
    assert res == hard_intersection(box1, box2)


def test_intersection_with_broadcasting() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )  # box_shape (2,2)
    box2 = BoxTensor(torch.tensor([[2, 0], [6, 2]]).float())  # box_shape (2,)
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]).float())
    assert res == hard_intersection(box1, box2)
    assert res == hard_intersection(box2, box1)


def test_intersection_module() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]).float())
    assert res == HardIntersection()(box1, box2)


def test_intersection_with_broadcasting_module() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )  # box_shape (2,2)
    box2 = BoxTensor(torch.tensor([[2, 0], [6, 2]]).float())  # box_shape (2,)
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]).float())
    assert res == HardIntersection()(box1, box2)
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )  # box_shape (2,2)
    box2 = BoxTensor(torch.tensor([[2, 0], [6, 2]]).float())  # box_shape (2,)
    assert res == HardIntersection()(box2, box1)


def test_intersection_with_broadcasting_module2() -> None:
    box1 = BoxTensor(
        torch.tensor([[[[1, 1], [4, 4]], [[2, 2], [5, 5]]]]).float()
    )  # box_shape (1, 2,2)
    assert box1.box_shape == (1, 2, 2)
    box2 = BoxTensor(
        torch.tensor([[[[3, 3], [7, 6]]], [[[1, 3], [3, 4]]]]).float()
    )
    assert box2.box_shape == (2, 1, 2)
    expected = BoxTensor(
        torch.tensor(
            [
                [[[3, 3], [4, 4]], [[3, 3], [5, 5]]],
                [[[1, 3], [3, 4]], [[2, 3], [3, 4]]],
            ]
        ).float()
    )
    assert expected == HardIntersection()(box1, box2)
    box1 = BoxTensor(
        torch.tensor([[[[1, 1], [4, 4]], [[2, 2], [5, 5]]]]).float()
    )  # box_shape (1, 2,2)
    assert box1.box_shape == (1, 2, 2)
    box2 = BoxTensor(
        torch.tensor([[[[3, 3], [7, 6]]], [[[1, 3], [3, 4]]]]).float()
    )
    assert box2.box_shape == (2, 1, 2)
    expected = BoxTensor(
        torch.tensor(
            [
                [[[3, 3], [4, 4]], [[3, 3], [5, 5]]],
                [[[1, 3], [3, 4]], [[2, 3], [3, 4]]],
            ]
        ).float()
    )
    assert expected == HardIntersection()(box2, box1)
