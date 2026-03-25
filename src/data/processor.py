import os
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    ETL pipeline for large-scale transportation datasets.
    Handles data cleaning, normalization, and time-series sequence generation.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads raw CSV data and logs basic statistics."""
        self.logger.info(f"Loading data from '{file_path}'")
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values using forward-fill then backward-fill.
        Avoids global mean imputation to prevent data leakage.
        """
        missing_before = df.isnull().sum().sum()
        # pandas 2.1+ replaced fillna(method=...) with .ffill()/.bfill()
        df = df.ffill().bfill()
        self.logger.info(f"Cleaned {missing_before} missing values.")
        return df

    def fit_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits a StandardScaler on all numeric columns and transforms the data.
        Saves the scaler so inference uses identical normalization.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df.copy()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        scaler_path = self.config.get("scaler_path", "models/scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"Scaler fitted and saved to '{scaler_path}'")
        return df

    def create_sequences(
        self, data: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts flat data into (Samples, Time-Steps, Features) arrays
        suitable for sequence models (LSTM, Transformer, etc.).
        """
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length, 0])  # first column as target
        return np.array(x), np.array(y)

    def process_pipeline(
        self, file_path: str
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Full pipeline: load -> clean -> normalize -> split into sequences.
        Returns (train_data, val_data) tuples ready for model.fit().
        """
        raw_df = self.load_data(file_path)
        clean_df = self.clean_data(raw_df)
        norm_df = self.fit_normalize(clean_df)

        # sequence_length is the number of time-steps per sample
        # (distinct from input_dim, which is the number of features)
        sequence_length = self.config["sequence_length"]
        X, y = self.create_sequences(norm_df.values, sequence_length)

        # Chronological train/val split — never shuffle time-series data
        val_split = self.config.get("val_split", 0.2)
        split_idx = int(len(X) * (1 - val_split))
        train_data = (X[:split_idx], y[:split_idx])
        val_data = (X[split_idx:], y[split_idx:])

        self.logger.info(
            f"Sequences: {len(X)} total | "
            f"train={len(train_data[0])}, val={len(val_data[0])}"
        )
        return train_data, val_data
