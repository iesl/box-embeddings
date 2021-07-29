from box_embeddings.modules.pooling import HardIntersectionBoxPooler
from box_embeddings.parameterizations.box_tensor import BoxTensor
import torch


def test_pooling_hard_intersection():
    box = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float())
    intersection_pooler = HardIntersectionBoxPooler(keepdim=True)
    mask = None
    res = intersection_pooler(box, mask)

    box_z = box.z
    box_Z = box.Z

    z = torch.max(box_z, dim=0, keepdim=True)[0]
    Z = torch.min(box_Z, dim=0, keepdim=True)[0]

    expected = box.like_this_from_zZ(z, Z)
    assert res == expected


def test_pooling_hard_intersection_with_mask():
    box = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float())
    intersection_pooler = HardIntersectionBoxPooler(keepdim=True)
    mask = torch.BoolTensor([[1, 0], [0, 1]])
    res = intersection_pooler(box, mask)

    box_z = box.z
    box_Z = box.Z
    box_z[mask] -= float("inf")
    box_Z[mask] += float("inf")
    z = torch.max(box_z, dim=0, keepdim=True)[0]
    Z = torch.min(box_Z, dim=0, keepdim=True)[0]

    expected = box.like_this_from_zZ(z, Z)
    assert res == expected
