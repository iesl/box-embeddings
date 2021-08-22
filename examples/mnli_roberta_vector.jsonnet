local test = std.extVar('TEST');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);
local transformer_model = "roberta-base";
local transformer_dim = 768;
local ff_hidden_1 = std.parseJson(std.extVar('ff_hidden_1'));
local ff_hidden_2 = std.parseJson(std.extVar('ff_hidden_2'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_activation = std.parseJson(std.extVar('ff_activation'));
local dropout = std.parseJson(std.extVar('dropout'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);

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
    "type": "mnli-vector-embeddings",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "premise_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": [ff_activation, "linear"],
      "dropout": [ff_dropout, 0],
    },
    "hypothesis_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": [ff_activation, "linear"],
      "dropout": [ff_dropout, 0],
    },
    "dropout": dropout,
    "namespace": "tags",
    "initializer": {
      "regexes": [
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    }
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
  }
}
