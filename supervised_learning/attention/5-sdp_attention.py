#!/usr/bin/env python3
"""attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Scaled Dot Product Attention"""
    # marmul
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # scale
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # output
    output = tf.matmul(weights, V)
    return output, weights
