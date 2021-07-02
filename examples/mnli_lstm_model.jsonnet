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
local vol_temp = std.parseJson(std.extVar('vol_temp'));
local box_reg_wt = std.parseJson(std.extVar('box_reg_wt'));
local box_tensor = 'mindelta_from_vector';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  "dataset_reader": {
    "type": "snli",
    tokenizer: {
      "type": 'whitespace',
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      }
    },
    "combine_input_fields": false,
    "collapse_labels": true
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_mismatched.jsonl",
  "model": {
    "type": "mnli",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
          "embedding_dim": 300,
          "trainable": true
        }
      }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 2,
        "bidirectional": true
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
      "input_dim": 300,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": [ff_activation, "linear"],
      "dropout": [ff_dropout, 0],
    },
    "hypothesis_feedforward": {
      "input_dim": 300,
      "num_layers": 2,
      "hidden_dims": [ff_hidden_1, ff_hidden_2],
      "activations": [ff_activation, "linear"],
      "dropout": [ff_dropout, 0],
    },
    "dropout": dropout,
    "namespace": "tags",
    "initializer": {
      "regexes": [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
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
