#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """encoder class"""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len, drop_rate=0.1):
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """call function"""
        seq_len = x.shape[1]
        # input word to embedding layer
        x = self.embedding(x)
        # scale embading layer by sqrt of dm to avoid gradient vanishing
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # add positional encoding sort of self attention
        x += self.positional_encoding[:seq_len]
        # dropout layer
        x = self.dropout(x, training=training)
        # build N blocks of encoder
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x
