"""Command-line entry point: train the MLP and write a Kaggle submission.

Example
-------
    python train.py \
        --train-csv data/raw/train.csv \
        --test-csv data/raw/test.csv \
        --submission data/processed/submission.csv \
        --epochs 10 --batch-size 42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from digit_recognizer import MLP, Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
        help="Path to the labelled training CSV (Kaggle 'train.csv').",
    )
    parser.add_argument(
        "--test-csv",
        default="data/raw/test.csv",
        help="Path to the unlabelled test CSV (Kaggle 'test.csv').",
    )
    parser.add_argument(
        "--submission",
        default="data/processed/submission.csv",
        help="Where to write the (ImageId, Label) submission file.",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Optional path to save the trained parameters as a .npy file.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=42, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    return parser.parse_args()


def load_training_data(train_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """Clean the training CSV and return ``(features, labels)`` arrays."""
    print("Preprocessing training data...")
    preprocessor = Preprocessor(train_csv)
    preprocessor.remove_duplicates()
    preprocessor.remove_null_values()
    labels, features = preprocessor.split_features_labels(label_names=["label"])

    labels = np.asarray(labels).T
    features = features.to_numpy()
    return features, labels


def main() -> None:
    args = parse_args()

    features, labels = load_training_data(args.train_csv)

    model = MLP(batch_size=args.batch_size, max_epoch=args.epochs, lr=args.lr)
    model.build_model()
    model.init_model()
    model.train(features, labels)

    if args.model_out:
        model.save_model(args.model_out)

    print("Predicting on the test set...")
    test_features = pd.read_csv(args.test_csv).to_numpy()
    predictions = model.evaluate(test_features)

    submission = pd.DataFrame(predictions.astype(int), columns=["Label"])
    submission.index += 1
    submission.index.name = "ImageId"
    submission.to_csv(args.submission)
    print(f"Wrote submission for {len(submission)} images to {args.submission}")


if __name__ == "__main__":
    main()
