# Review: Temporal Point Process on Predictive Business Process Monitoring

![img.png](flowchart.png)

The code is based on [1] and [2]. We thank the authors for their great work.

## List of TPP methods

We first conclude the recent research topics on deep TPPs as two main components:

- Encoding of history sequence

- Formulation of conditional intensity function

**TPP methods:**
| Methods | History Encoder | Intensity Function | Relational Discovery | Learning Approaches | Released codes |
|------------|-----------------|----------------------|----------------------------|---------------------|----------------------------------------------------------|
| [RMTPP](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)     | RNN | Gompertz | / | MLE with
SGD | https://github.com/musically-ut/tf_rmtpp                 |
| [ERTPP](https://arxiv.org/pdf/1705.08982.pdf)      | LSTM | Gaussian | / | MLE with
SGD | https://github.com/xiaoshuai09/Recurrent-Point-Process   |
| [CTLSTM](https://arxiv.org/pdf/1612.09328.pdf)     | CTLSTM | Exp-decay + softplus | / | MLE with
SGD | https://github.com/HMEIatJHU/neurawkes                   |
| [FNNPP](https://arxiv.org/pdf/1905.09690.pdf)      | LSTM | FNNIntegral | / | MLE with
SGD | https://github.com/omitakahiro/NeuralNetworkPointProcess |
| [LogNormMix](https://arxiv.org/pdf/1909.12127.pdf) | LSTM | Log-norm Mixture | / | MLE with
SGD | https://github.com/shchur/ifl-tpp                        |
| [SAHP](https://arxiv.org/pdf/1907.07561.pdf)       | Transformer | Exp-decay + softplus | Attention Matrix | MLE with
SGD | https://github.com/QiangAIResearcher/sahp_repo           |
| [THP](https://arxiv.org/pdf/2002.09291.pdf)        | Transformer | Linear + softplus | Structure learning | MLE with
SGD | https://github.com/SimiaoZuo/Transformer-Hawkes-Process  |
| [DGNPP](https://dl.acm.org/doi/pdf/10.1145/3442381.3450135)      | Transformer | Exp-decay + softplus | Bilevel
Structure learning | MLE with SGD | No available codes until now. |

## Installation

Requiring packages:

```
pip install -r requirements.txt
```

### Dataset

#### Converting from raw data to .pkl dataset
We evaluate on two real-world datasets: ``helpdesk`` and ``bpi_challenge_2012``. These datasets are generated and split by Rama et al. [1]
and located in ``./data/``. The data format is as follows:
``./data/{dataset_name}/``
The code for converting is also available in ``./scripts/``. For example:

```bash
python ./scripts/tax_dataloader.py --dataset BPI_Challenge_2012
```

#### Preprocessing the .pkl dataset
We preprocess the raw data to make it suitable for training TPP models. The preprocessing code is also available in ``./scripts/``. For example:

```bash
python ./scripts/generate_helpdesk_data.py
```

### Training

You can train the model with the following commands:

```bash
python main.py --dataset helpdesk
python main.py --dataset BPI_Challenge_2012
```

*Note: Whenever you want to train a new model, please delete the old checkpoint directory
at ``./experiment/{dataset_name}/{dataset_name}_debug``.*

The ``.yaml`` files consist following kwargs:

```
log_level: INFO

fold: The index of the fold data to use for training/val/test
time_pred_type: The type of time prediction: `interval` for the inter-event time, `timestamp` for the next timestamp

data:
  batch_size: The batch size for training
  dataset_dir: The processed dataset directory
  val_batch_size: The batch size for validation and test
  event_type_num: Number of the event types in the dataset. {'Helpdesk': 16, BPI_Challenge_2012': 37}

model:
  encoder_type: Used history encoder, chosen in [FNet, RNN, LSTM, GRU, Attention]
  intensity_type: Used intensity function, chosen in [LogNormMix, GomptMix, LogCauMix, ExpDecayMix, WeibMix, GaussianMix] and 
        [LogNormMixSingle, GomptMixSingle, LogCauMixSingle, ExpDecayMixSingle, WeibMixSingle, GaussianMixSingle, FNNIntegralSingle],
        where *Single means modeling the overall intensities
  time_embed_type: Time embedding, chosen in [Linear, Trigono]
  embed_dim: Embeded dimension
  lag_step: Predefined lag step, which is only used when intra_encoding is true
  atten_heads: Attention heads, only used in Attention encoder, must be a divisor of embed_dim.
  layer_num: The layers number in the encoder and history encoder
  dropout: Dropout ratio, must be in 0.0-1.0
  gumbel_tau: Initial temperature in Gumbel-max
  l1_lambda: Weight to control the sparsity of Granger causality graph
  use_prior_graph: Only be true when the ganger graph is given, chosen in [true, false]
  intra_encoding: Whether to use intra-type encoding,  chosen in [true, false]

train:
  epochs: Training epoches
  lr: Initial learning rate
  log_dir: Diretory for logger
  lr_decay_ratio: The decay ratio of learning rate
  max_grad_norm: Max gradient norm
  min_learning_rate: Min learning rate
  optimizer: The optimizer to use, chosen in [adam]
  patience: Epoch for early stopping 
  steps: Epoch numbers for learning rate decay. 
  test_every_n_epochs: 10
  experiment_name: and str, such as 'stackoverflow'
  delayed_grad_epoch: 10
  relation_inference: Whether to use graph discovery, chosen in [true, false],
        if false, but intra_encoding is true, the graph will be complete.
  
gpu: The GPU number to use for training

seed: Random Seed
```

## References

[1] E. Rama-Maneiro, J. Vidal, and M. Lama, “Deep Learning for Predictive Business Process Monitoring: Review and
Benchmark,” IEEE Transactions on Services Computing, pp. 1–1, 2022, doi: https://doi.org/10.1109/tsc.2021.3139807.

[2] H. Lin, C. Tan, L. Wu, Z. Gao, and S. Z. Li, “An Empirical Study: Extensive Deep Temporal Point Process,” arXiv.org,
2021. https://arxiv.org/abs/2110.09823