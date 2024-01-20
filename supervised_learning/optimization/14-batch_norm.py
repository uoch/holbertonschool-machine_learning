#!/usr/bin/env python3
"""Normalization Constants"""""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    create a layer of a neural network using batch normalization
    means , variances of the input data over the training data
    intialize the gamma and beta for the batch normalization layer
    tf.nn.batch_normalization
    apply the activation function to the normalized inputs
    """
    kernal = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n, activation=None,
        kernel_initializer=kernal)
    z = layer(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, epsilon)
    return activation(z_norm)
