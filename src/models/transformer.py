from typing import List

import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    """
    Transformer encoder block (Pre-LN variant) for time-series forecasting.
    Pre-LN (normalize before attention) is more stable than the original Post-LN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_size: int,
        ff_dim: int,
        rate: float = 0.1,
    ) -> None:
        super().__init__()
        # key_dim = per-head projection dimension (not total embed_dim)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=rate
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Pre-LN: normalize BEFORE attention sub-layer
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # residual

        # Pre-LN: normalize BEFORE FFN sub-layer
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output  # residual


class TokenAndPositionEmbedding(layers.Layer):
    """
    Projects raw features into embed_dim and adds learned positional embeddings.
    Without positional embeddings the Transformer treats all time-steps as equivalent,
    which defeats the purpose of using it on ordered time-series data.
    """

    def __init__(self, sequence_length: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_emb = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.token_proj = layers.Dense(embed_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len)
        return self.token_proj(x) + self.pos_emb(positions)


def build_transformer_model(
    input_shape: tuple,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    num_blocks: int,
    mlp_units: List[int],
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    Full Transformer encoder for traffic / energy consumption forecasting.

    Args:
        input_shape: (sequence_length, num_features)
        head_size:   Per-head projection dimension for MultiHeadAttention
        num_heads:   Number of parallel attention heads
        ff_dim:      Hidden units in the position-wise feed-forward layer
        num_blocks:  Number of stacked TransformerBlock layers
        mlp_units:   Hidden units in the final regression head
        dropout:     Dropout rate applied throughout
    """
    sequence_length, _ = input_shape
    embed_dim = head_size * num_heads  # standard convention

    inputs = tf.keras.Input(shape=input_shape)
    x = TokenAndPositionEmbedding(sequence_length, embed_dim)(inputs)

    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, head_size, ff_dim, dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="linear", name="prediction_output")(x)
    return tf.keras.Model(inputs, outputs, name="Transportation_Transformer")
