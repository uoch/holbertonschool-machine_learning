#!/usr/bin/env python3
"""regularization"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    dropout = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            name="layer")
    return dropout(layer(prev))
