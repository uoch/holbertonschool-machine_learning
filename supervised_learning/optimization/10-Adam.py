#!/usr/bin/env python3
"""Normalization Constants"""""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    """
    op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return op.minimize(loss)
