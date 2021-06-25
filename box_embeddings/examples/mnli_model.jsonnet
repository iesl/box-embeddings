local test = std.extVar('TEST');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);
local transformer_model = "roberta-base";
local transformer_dim = 768;
local ff_hidden_1 = std.parseJson(std.extVar('ff_hidden_1'));
local ff_hidden_2 = std.parseJson(std.extVar('ff_hidden_2'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local dropout = std.parseJson(std.extVar('dropout'));
local vol_temp = std.parseJson(std.extVar('vol_temp'));
local box_tensor = 'mindelta_from_vector';

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  "dataset_reader": {
    "type": "snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    },
    "combine_input_fields": false,
    "collapse_labels": true
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_mismatched.jsonl",
  "model": {
    "type": "mnli",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "box_factory": {
      "type": 'box_factory',
      "name": box_tensor,
    },
    "intersection": {
      "type": 'hard',
    },
    "volume": {
      "type": 'soft',
      "volume_temperature": vol_temp,
    },
    "premise_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": ["tanh", "linear"],
      "dropout": [ff_dropout, 0],
    },
    "hypothesis_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": ["tanh", "linear"],
      "dropout": [ff_dropout, 0],
    },
    "dropout": dropout,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 30,
    "cuda_device": std.parseInt(cuda_device),
    "patience": 20,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-6,
      "weight_decay": 0.1,
    },
    callbacks: [
      'track_epoch_callback',
      {
        type: 'tensorboard-custom',
        tensorboard_writer: {
          histogram_interval: 2000
        },
        model_outputs_to_log: ['premise_embedded_text', 'hypothesis_embedded_text', 'premise_embeddings', 'hypothesis_embeddings'],
      },
    ] + (if use_wandb then ['log_metrics_to_wandb'] else [])
  }
}
