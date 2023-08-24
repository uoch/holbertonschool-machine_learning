#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """return a model with updated weights and biases
    which is gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
