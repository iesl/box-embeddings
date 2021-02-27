from box_embeddings.parameterizations.box_tensor import BoxTensor, BoxFactory
from box_embeddings.modules.embedding import BoxEmbedding
import torch


def test_simple_creation() -> None:
    embedding = BoxEmbedding(5, 10)
    assert embedding.num_embeddings == 5
    assert (
        embedding.embedding_dim
        == 10 * embedding.box_factory.box_subclass.w2z_ratio
    )
    assert embedding.box_factory.name == 'mindelta_from_vector'


def test_creation_with_box_factory() -> None:
    box_factory = BoxFactory('sigmoid_from_vector')
    embedding = BoxEmbedding(5, 10, box_factory=box_factory)
    assert embedding.box_factory.name == 'sigmoid_from_vector'


def test_forward() -> None:
    embedding = BoxEmbedding(5, 10)
    box_factory = BoxFactory('mindelta_from_vector')
    W = embedding.weight
    assert W.shape == (5, 20)
    box_tensor = box_factory(W)
    box_tensor_direct = box_tensor[2:4, ...]
    inputs = torch.tensor([2, 3])
    box_tensor_from_emb = embedding.forward(inputs)

    assert (box_tensor_direct.z == box_tensor_from_emb.z).all()
    assert (box_tensor_direct.Z == box_tensor_from_emb.Z).all()
