import torch
from box_embeddings.modules.volume.volume import HardVolume
from box_embeddings.parameterizations.box_tensor import BoxTensor


def test_volume() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    volume_layer = HardVolume(log_scale=False)
    expected1 = torch.tensor([8, 4]).float()
    expected2 = torch.tensor([8, 2]).float()
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert torch.allclose(res1, expected1)
    assert torch.allclose(res2, expected2)


def test_log_volume() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    volume_layer = HardVolume(log_scale=True)
    expected1 = torch.tensor([2.07944, 1.3862]).float()
    expected2 = torch.tensor([2.07944, 0.69314]).float()
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert torch.allclose(res1, expected1, rtol=1e-4)
    assert torch.allclose(res2, expected2, rtol=1e-4)
