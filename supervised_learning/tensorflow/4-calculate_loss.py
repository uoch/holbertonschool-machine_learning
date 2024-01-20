#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the loss for a given y_pred"""
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred,)
    return loss
