local transformer_model = "roberta-base";
local transformer_dim = 768;

{
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
      "name": 'mindelta_from_vector',
    },
    "intersection": {
      "type": 'hard',
    },
    "volume": {
      "type": 'soft',
      "volume_temperature": 20,
    },
    "premise_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [500, 200],
      "activations": "tanh",
      "dropout": 0.2,
    },
    "hypothesis_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [500, 200],
      "activations": "tanh",
      "dropout": 0.2,
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 16
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-6,
      "weight_decay": 0.1,
    }
  }
}
