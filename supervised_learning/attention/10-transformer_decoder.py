#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """decoder class"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """constructor"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """call function"""
        # get output shape
        seq_len = x.shape[1]
        # output word to embedding layer
        x = self.embedding(x)
        # scale embading layer by sqrt of dm to avoid gradient vanishing
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # add positional encoding sort of self attention
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        # build N blocks of decoder
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x
