"""Neural network building blocks implemented from scratch with NumPy.

Each layer follows a common ``forward`` / ``backward`` convention so that they
can be composed into arbitrary feed-forward networks:

* ``forward(x)`` caches the inputs needed by the backward pass and returns the
  layer output.
* ``backward(top_diff)`` receives the gradient of the loss with respect to the
  layer output and returns the gradient with respect to the layer input.
"""

from __future__ import annotations

import numpy as np


class FullyConnectedLayer:
    """A dense (affine) layer computing ``y = x @ W + b``."""

    def __init__(self, num_input: int, num_output: int) -> None:
        self.num_input = num_input
        self.num_output = num_output
        self.weight: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def init_param(self, std: float = 0.01) -> None:
        """Initialise weights from a Gaussian and biases to zero."""
        self.weight = np.random.normal(
            loc=0.0, scale=std, size=(self.num_input, self.num_output)
        )
        self.bias = np.zeros((1, self.num_output))

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Compute the affine transform and cache the input for backprop."""
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff: np.ndarray) -> np.ndarray:
        """Accumulate parameter gradients and propagate the input gradient."""
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr: float) -> None:
        """Apply a single vanilla SGD step to the parameters."""
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight: np.ndarray, bias: np.ndarray) -> None:
        """Load pre-trained parameters, validating their shapes."""
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(weight, bias)`` tuple."""
        return self.weight, self.bias


class ReLULayer:
    """Rectified Linear Unit activation: ``y = max(0, x)``."""

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def backward(self, top_diff: np.ndarray) -> np.ndarray:
        bottom_diff = top_diff.copy()
        bottom_diff[self.input < 0] = 0
        return bottom_diff


class SoftmaxLossLayer:
    """Softmax activation combined with a cross-entropy loss."""

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Return class probabilities using a numerically stable softmax."""
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label: np.ndarray) -> float:
        """Compute the mean cross-entropy loss for a batch of integer labels."""
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self) -> np.ndarray:
        """Return the gradient of the cross-entropy loss w.r.t. the logits."""
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
