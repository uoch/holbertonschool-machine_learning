#!/usr/bin/env python3
"""builds a neural network with the Keras library"""""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimiz for a keras model with categorical crossentropy"""
    op = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    opt = network.compile(
        optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
    return None
