import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf


class Trainer:
    """
    Encapsulates the full training loop with industry-standard callbacks.
    """

    def __init__(self, model: tf.keras.Model, config: dict) -> None:
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._callbacks = self._build_callbacks()

    def _build_callbacks(self) -> list:
        checkpoint_path = os.path.join(
            self.config["paths"]["checkpoint_dir"],
            # .keras format replaces deprecated .h5 in TF 2.15+
            "best_model.keras",
        )
        es_cfg = self.config["training"]["early_stopping"]

        return [
            # 1. Early stopping — prevents overfitting, saves compute
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                restore_best_weights=es_cfg["restore_best_weights"],
                verbose=1,
            ),
            # 2. Save the best checkpoint (not just the final epoch)
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            # 3. Reduce LR when val_loss plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            # 4. TensorBoard for experiment tracking and comparison
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config["paths"]["log_dir"],
                histogram_freq=1,
                update_freq="epoch",
            ),
        ]

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
    ) -> tf.keras.callbacks.History:
        """Compiles the model, runs training, and returns the History object."""
        train_cfg = self.config["training"]
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_cfg["optimizer"]["lr"]
        )
        self.model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        self.logger.info(
            f"Training: up to {train_cfg['epochs']} epochs, "
            f"batch_size={train_cfg['batch_size']}"
        )
        history = self.model.fit(
            train_data[0],
            train_data[1],
            validation_data=val_data,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            callbacks=self._callbacks,
            verbose=1,
        )
        self.logger.info("Training complete.")
        return history
