from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.trainer import (
    TensorBoardCallback,
    TrainerCallback,
    GradientDescentTrainer,
    TensorDict,
    TensorBoardWriter,
)
import warnings
import logging

logger = logging.getLogger(__name__)


@TrainerCallback.register("tensorboard-custom")
class CustomTensorBoardCallback(TensorBoardCallback):
    def __init__(
        self,
        serialization_dir: str,
        tensorboard_writer: Lazy[TensorBoardWriter] = Lazy(TensorBoardWriter),
        model_outputs_to_log: List[str] = None,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            tensorboard_writer=tensorboard_writer,
        )
        self._model_outputs_to_log = model_outputs_to_log or []
        self._warned_about_missing_keys = False

    def _warn_about_missing_keys(
        self, model_outputs: List[Dict[str, Any]]
    ) -> None:

        if not self._warned_about_missing_keys:
            for key in self._model_outputs_to_log:
                if key not in model_outputs[0]:
                    logger.warning(f"Key {key} missing in model outputs.")
            self._warned_about_missing_keys = True

        return

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # do everything as the parent does
        super().on_batch(
            trainer,
            batch_inputs,
            batch_outputs,
            batch_metrics,
            epoch,
            batch_number,
            is_training,
            is_primary=is_primary,
            batch_grad_norm=batch_grad_norm,
            **kwargs,
        )
        assert len(batch_outputs) == 1, "Gradient accumulation not supported"
        self._warn_about_missing_keys(batch_outputs)
        if self._tensorboard.should_log_histograms_this_batch():
            for key in self._model_outputs_to_log:
                value = batch_outputs[0].get(key, None)

                if value is not None:
                    if is_training:
                        self._tensorboard.add_train_histogram(  # type: ignore
                            "model_outputs/" + key, value
                        )
