from typing import List, Tuple, Union, Dict, Any, Optional
import tensorflow as tf
from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxFactory,
    TFBoxTensor,
)
from box_embeddings.initializations.tf_initializer import TFBoxInitializer
from box_embeddings.initializations.tf_uniform_boxes import (
    TFUniformBoxInitializer,
)


class TFBoxEmbedding(tf.keras.layers.Embedding):
    """Embedding which returns boxes instead of vectors"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        box_factory: TFBoxFactory = None,
        box_initializer: TFBoxInitializer = None,
        **kwargs: Any,
    ) -> None:
        box_factory = box_factory or TFBoxFactory("mindelta_from_vector")
        super().__init__(
            num_embeddings,
            embedding_dim * box_factory.box_subclass.w2z_ratio,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = self.output_dim
        self.box_factory = box_factory

        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            dtype=tf.float64,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False,
        )

        if box_initializer is None:
            box_initializer = TFUniformBoxInitializer(
                dimensions=embedding_dim,  # here dim is box dim
                num_boxes=int(self.input_dim),  # type: ignore
                box_type_factory=self.box_factory,
            )
        self.built = True
        box_initializer(self.embeddings)

    def call(self, inputs: tf.Tensor) -> TFBoxTensor:
        emb = super().call(inputs)
        box_emb = self.box_factory(emb)

        return box_emb

    @property
    def all_boxes(self) -> TFBoxTensor:
        all_ = self.box_factory(self.embeddings)  # type:ignore

        return all_

    def get_bounding_box(self) -> TFBoxTensor:
        all_ = self.all_boxes
        z = all_.z  # shape = (num_embeddings, box_embedding_dim)
        Z = all_.Z
        z_min, _ = z.min(dim=0)
        Z_max, _ = Z.max(dim=0)

        return self.box_factory.box_subclass.from_zZ(z_min, Z_max)
