import torch
from box_embeddings.parameterizations.box_tensor import BoxFactory, BoxTensor
from box_embeddings.initializations.initializer import BoxInitializer
from box_embeddings.initializations.uniform_boxes import UniformBoxInitializer


class BoxEmbedding(torch.nn.Embedding):
    """Embedding which returns boxes instead of vectors"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        box_factory: BoxFactory = None,
        box_initializer: BoxInitializer = None,
        **kwargs,
    ) -> None:
        box_factory = box_factory or BoxFactory("mindelta_from_vector")
        super().__init__(
            num_embeddings,
            embedding_dim * box_factory.box_subclass.w2z_ratio,
            **kwargs,
        )

        self.box_factory = box_factory

        if box_initializer is None:
            box_initializer = UniformBoxInitializer(
                dimensions=embedding_dim,  # here dim is box dim
                num_boxes=int(self.weight.shape[0]),  # type: ignore
                box_type_factory=self.box_factory,
            )

        box_initializer(self.weight)

    def forward(self, inputs: torch.Tensor) -> BoxTensor:  # type:ignore
        emb = super().forward(inputs)  # shape (..., self.box_embedding_dim*2)
        box_emb = self.box_factory(emb)

        return box_emb

    @property
    def all_boxes(self) -> BoxTensor:
        all_ = self.box_factory(self.weight)  # type:ignore

        return all_

    def get_bounding_box(self) -> BoxTensor:
        all_ = self.all_boxes
        z = all_.z  # shape = (num_embeddings, box_embedding_dim)
        Z = all_.Z
        z_min, _ = z.min(dim=0)
        Z_max, _ = Z.max(dim=0)

        return self.box_factory.box_subclass.from_zZ(z_min, Z_max)
