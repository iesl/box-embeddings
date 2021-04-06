from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxTensor,
    TFBoxFactory,
)
from box_embeddings.modules.tf_embedding import TFBoxEmbedding
import tensorflow as tf


def test_simple_creation() -> None:
    embedding = TFBoxEmbedding(5, 10)
    assert embedding.num_embeddings == 5
    assert (
        embedding.embedding_dim
        == 10 * embedding.box_factory.box_subclass.w2z_ratio
    )
    assert embedding.box_factory.name == 'mindelta_from_vector'


def test_creation_with_box_factory() -> None:
    box_factory = TFBoxFactory('sigmoid_from_vector')
    embedding = TFBoxEmbedding(5, 10, box_factory=box_factory)
    assert embedding.box_factory.name == 'sigmoid_from_vector'


def test_forward() -> None:
    embedding = TFBoxEmbedding(5, 10)
    box_factory = TFBoxFactory('mindelta_from_vector')
    W = embedding.embeddings
    assert W.shape == (5, 20)
    box_tensor = box_factory(W)
    box_tensor_direct = box_tensor.data.numpy()[2:4, ...]
    inputs = tf.Variable([2, 3])
    box_tensor_from_emb = embedding(inputs)

    assert (box_tensor_direct == box_tensor_from_emb.data.numpy()).all()


def test_all_boxes() -> None:
    embedding = TFBoxEmbedding(5, 10)
    box_factory = TFBoxFactory('mindelta_from_vector')
    W = embedding.embeddings
    box_tensor = box_factory(W)
    box_tensor_from_emb = embedding.all_boxes
    assert (box_tensor.data.numpy() == box_tensor_from_emb.data.numpy()).all()
