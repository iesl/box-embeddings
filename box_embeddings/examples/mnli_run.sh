#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/box-embeddings-%j.out
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
export ff_hidden_1=500
export ff_hidden_2=200
export ff_dropout=0.2
export dropout=0.1
export vol_temp=20
# export ff_activation=softplus
# export inf_optim=sgd
# export sample_picker=lastn

# wandb_allennlp --subcommand=train --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structure_prediction_baselines --wandb_entity score-based-learning --wandb_tags=baselines,as_reported

wandb_allennlp --subcommand=train \
	--config_file=mnli_model.jsonnet \
	--include-package=box_embeddings \
	--wandb_run_name=box-embeddings_mnli \
	--wandb_project box-embeddings \
	--wandb_entity purujit \
	--wandb_tags=mnli
