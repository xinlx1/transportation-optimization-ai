from typing import Any, Dict

import tensorflow as tf


def build_optimization_model(config: Dict[str, Any]) -> tf.keras.Model:
    """
    Factory function to build a Dense feed-forward model via the functional API.

    Using the functional API (vs Sequential + Input layer) gives:
    - A properly defined input signature for TF Serving / SavedModel export
    - Support for multi-input / multi-output extensions later
    """
    input_shape = (config["sequence_length"], config["input_dim"])
    inputs = tf.keras.Input(shape=input_shape, name="input_features")
    x = tf.keras.layers.Flatten(name="sequence_flattener")(inputs)
    for i, units in enumerate(config["hidden_units"]):
        x = tf.keras.layers.Dense(
            units,
            kernel_initializer="he_normal",  # recommended for ReLU networks
            use_bias=False,  # bias is redundant before BatchNorm
            name=f"fc_{i + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{i + 1}")(x)
        x = tf.keras.layers.Dropout(
            config["dropout_rate"], name=f"dropout_{i + 1}"
        )(x)

    outputs = tf.keras.layers.Dense(
        1, activation="linear", name="prediction_output"
    )(x)

    return tf.keras.Model(
        inputs, outputs, name="Transportation_Optimization_Model"
    )
