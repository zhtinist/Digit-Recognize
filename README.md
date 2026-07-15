# Digit Recognizer

A multi-layer perceptron (MLP) for handwritten digit classification on the
[Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
dataset (MNIST), implemented **from scratch in NumPy** — including the forward
pass, backpropagation, and mini-batch stochastic gradient descent. No deep
learning framework is used.

## Features

- Fully-connected network: `784 → 256 → 128 → 64 → 10` with ReLU activations
  and a softmax cross-entropy output.
- Hand-written forward/backward passes for every layer.
- Numerically stable softmax and cross-entropy loss.
- Simple pandas-based preprocessing (deduplication, null removal, feature/label
  split).
- A command-line trainer that produces a Kaggle-ready submission file.

Reference run: **~99.5% training accuracy** after 10 epochs (`batch_size=42`).

## Project layout

```
.
├── digit_recognizer/        # importable package
│   ├── __init__.py
│   ├── layers.py            # FullyConnected / ReLU / SoftmaxLoss layers
│   ├── mlp.py               # MLP model: build, train, evaluate, save/load
│   └── preprocess.py        # Preprocessor: cleaning & feature/label split
├── notebooks/
│   └── digit_recognizer.ipynb   # exploratory walkthrough
├── data/                    # datasets (not committed — see data/README.md)
├── train.py                 # CLI entry point
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle and place the CSVs under `data/raw/`
   (see [`data/README.md`](data/README.md)).
2. Train the model and generate a submission:

```bash
python train.py \
    --train-csv data/raw/train.csv \
    --test-csv data/raw/test.csv \
    --submission data/processed/submission.csv \
    --epochs 10 --batch-size 42 --lr 0.01
```

The submission file has the columns `ImageId,Label` expected by Kaggle.

### Library usage

```python
import numpy as np
from digit_recognizer import MLP, Preprocessor

pre = Preprocessor("data/raw/train.csv")
pre.remove_duplicates()
pre.remove_null_values()
labels, features = pre.split_features_labels(["label"])

model = MLP(batch_size=42, max_epoch=10, lr=0.01)
model.build_model()
model.init_model()
model.train(features.to_numpy(), np.asarray(labels).T)
```

## How it works

Each layer implements a `forward` / `backward` pair:

- **FullyConnectedLayer** — affine transform `y = xW + b`; the backward pass
  accumulates `dW`, `db` and returns the input gradient.
- **ReLULayer** — `max(0, x)`; gradients are zeroed where the input was
  negative.
- **SoftmaxLossLayer** — stable softmax plus mean cross-entropy loss; its
  backward pass returns `(prob - onehot) / batch_size`.

`MLP.train` chains these layers, running forward → loss → backward → SGD update
per mini-batch and logging loss and accuracy each epoch.

## License

Released under the [MIT License](LICENSE).
