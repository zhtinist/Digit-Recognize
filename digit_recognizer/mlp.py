"""A configurable multi-layer perceptron (MLP) classifier.

The network is a fully-connected feed-forward architecture with three hidden
ReLU layers and a softmax cross-entropy output, trained with mini-batch
stochastic gradient descent. Everything is implemented on top of NumPy using
the primitives defined in :mod:`digit_recognizer.layers`.
"""

from __future__ import annotations

import numpy as np

from digit_recognizer.layers import (
    FullyConnectedLayer,
    ReLULayer,
    SoftmaxLossLayer,
)


class MLP:
    """Four-layer fully-connected classifier (784 -> 256 -> 128 -> 64 -> 10)."""

    def __init__(
        self,
        batch_size: int = 30,
        input_size: int = 784,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        out_classes: int = 10,
        lr: float = 0.01,
        max_epoch: int = 30,
        print_iter: int = 100,
    ) -> None:
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def shuffle_data(self) -> None:
        """Shuffle the training set in place between epochs."""
        np.random.shuffle(self.train_data)

    def build_model(self) -> None:
        """Instantiate the layers that make up the network."""
        print("Building multi-layer perceptron model...")
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReLULayer()
        self.fc4 = FullyConnectedLayer(self.hidden3, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def init_model(self) -> None:
        """Randomly initialise the parameters of every trainable layer."""
        print("Initializing parameters of each layer in MLP...")
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir: str) -> None:
        """Load parameters previously written by :meth:`save_model`."""
        print(f"Loading parameters from file {param_dir}")
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params["w1"], params["b1"])
        self.fc2.load_param(params["w2"], params["b2"])
        self.fc3.load_param(params["w3"], params["b3"])
        self.fc4.load_param(params["w4"], params["b4"])

    def save_model(self, param_dir: str) -> None:
        """Serialise all layer parameters to a ``.npy`` file."""
        print(f"Saving parameters to file {param_dir}")
        params = {}
        params["w1"], params["b1"] = self.fc1.save_param()
        params["w2"], params["b2"] = self.fc2.save_param()
        params["w3"], params["b3"] = self.fc3.save_param()
        params["w4"], params["b4"] = self.fc4.save_param()
        np.save(param_dir, params)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Run a forward pass and return the class probabilities."""
        h1 = self.relu1.forward(self.fc1.forward(input))
        h2 = self.relu2.forward(self.fc2.forward(h1))
        h3 = self.relu3.forward(self.fc3.forward(h2))
        logits = self.fc4.forward(h3)
        prob = self.softmax.forward(logits)
        return prob

    def backward(self) -> None:
        """Run a full backward pass, populating every layer's gradients."""
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        self.fc1.backward(dh1)

    def update(self, lr: float) -> None:
        """Apply an SGD update to every trainable layer."""
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the network with mini-batch SGD, logging per-epoch metrics."""
        self.train_data = np.concatenate((features, labels), axis=1)
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        print("Start training...")
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            acc_num = 0
            total_loss = 0.0
            for idx_batch in range(max_batch):
                batch = self.train_data[
                    idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size
                ]
                batch_images = batch[:, :-1]
                batch_labels = batch[:, -1]

                prob = self.forward(batch_images)
                total_loss += self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)

                pred_labels = np.argmax(prob, axis=1)
                acc_num += np.sum(pred_labels == batch_labels)

            avg_loss = total_loss / max_batch
            accuracy = acc_num / (max_batch * self.batch_size)
            print(
                f"Epoch {idx_epoch + 1}: Loss = {avg_loss:.6f}, "
                f"Accuracy = {accuracy:.6f}"
            )

    def evaluate(self, feature: np.ndarray) -> np.ndarray:
        """Return the predicted class for each row of ``feature``."""
        self.test_data = feature
        pred_results = np.zeros(self.test_data.shape[0])
        for idx in range(self.test_data.shape[0]):
            image = self.test_data[idx : idx + 1, :]
            prob = self.forward(image)
            pred_results[idx] = int(np.argmax(prob, axis=1)[0])
        return pred_results
