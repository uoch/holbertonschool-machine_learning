#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """transformer class"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input,
                 max_seq_target, drop_rate=0.1):
        """constructor"""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """call function"""
        enc_output = self.encoder(inputs, training, encoder_mask)
        # enc_output will be input to decoder
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.final_layer(dec_output)
        return final_output
