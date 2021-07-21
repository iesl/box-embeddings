from box_embeddings.modules.regularization import L2SideBoxRegularizer
from box_embeddings.parameterizations.box_tensor import BoxTensor
import torch

eps = 1e-23


def test_l2_side_regularizer():
    box = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float())
    regularizer = L2SideBoxRegularizer(weight=0.1)

    z = box.z  # (..., box_dim)
    Z = box.Z  # (..., box_dim)

    expected = 0.1 * torch.sum((Z - z) ** 2)
    res = regularizer(box)
    assert res == expected


def test_l2_side_regularizer_log():
    box = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float())
    regularizer = L2SideBoxRegularizer(weight=0.1, log_scale=True)

    z = box.z  # (..., box_dim)
    Z = box.Z  # (..., box_dim)

    expected = 0.1 * torch.sum(torch.log(torch.abs(Z - z) + eps))
    res = regularizer(box)
    assert res == expected
