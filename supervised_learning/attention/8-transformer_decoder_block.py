#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """call function"""
        # masked multi-head attention block
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        # add and norm block
        out1 = self.layer_norm1(x + attn1)
        # multi-head attention block linked to encoder output
        attn2, _ = self.mha2(
            out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        # add and norm block
        out2 = self.layer_norm2(out1 + attn2)
        # feed forward block (dense)
        ffn = tf.keras.Sequential([
            self.dense_hidden,
            self.dense_output
        ])
        ffn_output = ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        # add and norm block
        output = self.layer_norm3(out2 + ffn_output)
        return output
