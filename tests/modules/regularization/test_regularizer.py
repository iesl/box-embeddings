from box_embeddings.modules.regularization import BoxRegularizer
from box_embeddings.parameterizations.box_tensor import BoxTensor
import torch


def test_regularizer():
    box = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float())
    regularizer = BoxRegularizer(weight=0.1)

    expected = 0.0
    res = regularizer(box)
    assert res == expected
