#!/usr/bin/env python3
"""tensorflow project"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Create a dense layer with a specified number of units and activation function.

    Args:
        prev: Previous layer or input tensor.
        n: Number of units in the layer.
        activation: Activation function to be applied to the layer.

    Returns:
        Layer with the specified number of units and activation function applied.
    bias are automatically computed by tensorflow.keras"""
    # Using FAN_AVG initializer for the weights
    weights_initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    # Create the dense layer with the specified activation and units
    dense_layer = tf.keras.layers.Dense(
        units=n, activation=activation, kernel_initializer=weights_initializer)

    # Apply the layer to the previous layer or input tensor
    layer = dense_layer(prev)

    return layer
