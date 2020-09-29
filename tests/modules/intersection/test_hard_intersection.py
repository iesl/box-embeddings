from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection.hard_intersection import (
    hard_intersection,
    HardIntersection,
)
import torch


def test_intersection() -> None:
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]))
    box2 = BoxTensor(torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]))
    assert (
        res.data.numpy() == hard_intersection(box1, box2).data.numpy()
    ).all()
