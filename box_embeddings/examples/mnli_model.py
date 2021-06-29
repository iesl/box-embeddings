from typing import Dict, Any, Optional, Union

import torch
from allennlp.data import TextFieldTensors, TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    TextFieldEmbedder,
    FeedForward,
    Seq2VecEncoder,
    Seq2SeqEncoder,
)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy

from box_embeddings.common.utils import log1mexp
from box_embeddings.modules.regularization import BoxRegularizer
from box_embeddings.modules.volume._volume import _Volume
from box_embeddings.modules.intersection._intersection import _Intersection
from box_embeddings.parameterizations.box_tensor import BoxFactory
import logging

logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(
    logging.INFO
)


@Model.register("mnli")
class MNLIModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Union[Seq2VecEncoder, Seq2SeqEncoder],
        box_factory: BoxFactory,
        intersection: _Intersection,
        volume: _Volume,
        premise_feedforward: FeedForward,
        hypothesis_feedforward: FeedForward,
        dropout: Optional[float] = None,
        box_regularizer: Optional[BoxRegularizer] = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab, regularizer=regularizer)  # type:ignore
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._box_factory = box_factory
        self._box_intersection = intersection
        self._box_volume = volume
        self._premise_feedforward = premise_feedforward
        self._hypothesis_feedforward = hypothesis_feedforward
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        if box_regularizer:
            self._box_regularizer = box_regularizer
        else:
            self._box_regularizer = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(
                namespace=self._label_namespace
            )
        # self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = torch.nn.NLLLoss()
        self._accuracy = BooleanAccuracy()
        if initializer is not None:
            initializer(self)

    def forward(  # type: ignore
        self,
        premise: TextFieldTensors,
        hypothesis: TextFieldTensors,
        label: torch.IntTensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        premise_embedded_text = self._text_field_embedder(premise)
        hypothesis_embedded_text = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise)
        hypothesis_mask = get_text_field_mask(hypothesis)

        premise_embedded_text = self._encoder(
            premise_embedded_text, mask=premise_mask
        )
        hypothesis_embedded_text = self._encoder(
            hypothesis_embedded_text, mask=hypothesis_mask
        )

        activations = {
            'premise_embedded_text': premise_embedded_text,
            'hypothesis_embedded_text': hypothesis_embedded_text,
        }
        if self._dropout:
            premise_embedded_text = self._dropout(premise_embedded_text)
            hypothesis_embedded_text = self._dropout(hypothesis_embedded_text)

        premise_embeddings = self._premise_feedforward(premise_embedded_text)
        hypothesis_embeddings = self._hypothesis_feedforward(
            hypothesis_embedded_text
        )

        activations['premise_embeddings'] = premise_embeddings
        activations['hypothesis_embeddings'] = hypothesis_embeddings
        premise_box = self._box_factory(premise_embeddings)
        hypothesis_box = self._box_factory(hypothesis_embeddings)

        y_prob = self._box_volume(
            self._box_intersection(premise_box, hypothesis_box)
        ) - self._box_volume(premise_box)
        output_dict = {"y_prob": y_prob}
        # output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(
                torch.stack((y_prob, log1mexp(y_prob)), dim=-1),
                label.long().view(-1),
            ) + self._box_regularizer(
                self._box_intersection(premise_box, hypothesis_box)
            )
            output_dict["loss"] = loss
            y_pred = 1 - torch.round(torch.exp(y_prob.detach()))
            self._accuracy(y_pred, label)

        output_dict.update(activations)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
