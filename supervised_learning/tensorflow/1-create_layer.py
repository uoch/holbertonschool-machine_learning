#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a dense layer with a specified number of units and activation func.

    Args:
        prev: Previous layer or input tensor.
        n: Number of units in the layer.
        activation: Activation function to be applied to the layer.

    Returns:
        Layer with the specified numb of units and activation function applied.
    bias are automatically computed by tensorflow."""
    # Using FAN_AVG initializer for the weights
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the dense layer with the specified activation and units
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=weights_initializer)

    # Apply the layer to the previous layer or input tensor

    return layer(prev)
