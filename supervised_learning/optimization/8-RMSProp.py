#!/usr/bin/env python3
"""Normalization Constants"""""
import tensorflow.compat.v1 as tf

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    """
    op = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return op.minimize(loss)