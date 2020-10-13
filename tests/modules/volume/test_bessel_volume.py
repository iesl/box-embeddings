import torch
from box_embeddings.modules.volume.bessel_volume import BesselApproxVolume
from box_embeddings.parameterizations.box_tensor import BoxTensor


def test_volume() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    volume_layer = BesselApproxVolume(log_scale=False)
    expected1 = torch.tensor([3.490526, 1.4467278]).float()
    expected2 = torch.tensor([3.490526, 0.7444129]).float()
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert torch.allclose(res1, expected1, rtol=1e-4)
    assert torch.allclose(res2, expected2, rtol=1e-4)


def test_log_volume() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    volume_layer = BesselApproxVolume(log_scale=True)
    expected1 = torch.tensor([1.25004, 0.36924]).float()
    expected2 = torch.tensor([1.25004, -0.29517]).float()
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert torch.allclose(res1, expected1, rtol=1e-4)
    assert torch.allclose(res2, expected2, rtol=1e-4)
