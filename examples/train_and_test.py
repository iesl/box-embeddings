"""This is essentially the same command as `allennlp.train` but it allows to run evaluation on test set after training is completed and logs the results to wandb"""
from typing import Dict, Any
from allennlp.commands.train import TrainModel
from allennlp.commands import Subcommand
from allennlp.common import util as common_util
from allennlp.training import util as training_util
import os
import logging

logger = logging.getLogger(__name__)


@TrainModel.register(
    "train_test_log_to_wandb", constructor="from_partial_objects"
)  # same construction pipeline as parent
class TrainTestAndLogToWandb(TrainModel):
    """Does the same thing as `allennlp.commands.train.TrainModel` but
    logs final metrics to `wandb` summary.

    To use this class, add the following to the top-level config.

    .. code-block:: JSON

        {
           type: 'train_test_log_to_wandb',
           evaluate_on_test: true,
           dataset_reader: { ...},
           model: {...},
           ...
        }

    """

    def finish(self, metrics: Dict[str, Any]) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        import wandb  # noqa

        if self.evaluation_data_loader is not None and self.evaluate_on_test:
            logger.info(
                "The model will be evaluated using the best epoch weights."
            )
            test_metrics = training_util.evaluate(
                self.model,
                self.evaluation_data_loader,  # type:ignore
                cuda_device=self.trainer.cuda_device,  # type: ignore
                batch_weight_key=self.batch_weight_key,
            )

            for key, value in test_metrics.items():
                metrics["test_" + key] = value
        elif self.evaluation_data_loader is not None:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"),
            metrics,
            log=True,
        )
        # update the summary with all metrics
        wandb.run.summary.update(metrics)
