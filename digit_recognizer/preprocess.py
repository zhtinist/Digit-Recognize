"""Lightweight tabular pre-processing utilities built on pandas."""

from __future__ import annotations

import pandas as pd


class Preprocessor:
    """Load a CSV dataset and expose common cleaning and splitting steps."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.labels: list[pd.Series] = []
        self.features = self.data

    def remove_duplicates(self) -> pd.DataFrame:
        """Drop duplicate rows from the dataset."""
        self.data = self.data.drop_duplicates()
        return self.data

    def remove_null_values(self) -> pd.DataFrame:
        """Drop rows that contain any missing values."""
        self.data = self.data.dropna()
        return self.data

    def save(self, output_path: str) -> None:
        """Write the current dataset to ``output_path`` as CSV."""
        self.data.to_csv(output_path, index=False)

    def split_features_labels(
        self, label_names: list[str]
    ) -> tuple[list[pd.Series], pd.DataFrame]:
        """Split the given label columns off from the feature columns."""
        for label_name in label_names:
            self.labels.append(self.data[label_name])
            self.features = self.features.drop(label_name, axis=1)
        return self.labels, self.features
