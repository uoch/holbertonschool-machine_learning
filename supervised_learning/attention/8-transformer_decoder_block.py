#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Init the class
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden unit
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """call function"""
        # masked multi-head attention block
        attention, _ = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention)
        # add and norm block
        out1 = self.layernorm1(attention + x)
        # multi-head attention block
        attention2, _ = self.mha2(out1, enc_output, enc_output, padding_mask)
        attention2 = self.dropout2(attention2)
        # add and norm block
        out2 = self.layernorm2(attention2 + out1)
        # feed forward block
        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn)
        # add and norm block
        out3 = self.layernorm3(ffn + out2)
        return out3
