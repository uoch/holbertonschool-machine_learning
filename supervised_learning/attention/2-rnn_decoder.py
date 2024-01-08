#!/usr/bin/env python3
"""attention"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """rnn encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor
        embedding is an int representing the dimensionality of the embedding
        vector
        gru recurrent layer to encode the input sequence"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab,
                                                   embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """calls encoder
        use embedding layer to create an embedding vector
        then pass the embedding vector to the GRU layer
        Returns: outputs, hidden"""
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, hidden = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
