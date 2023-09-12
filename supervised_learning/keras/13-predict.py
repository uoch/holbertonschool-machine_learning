#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ predicts a neural network"""
    return network.predict(data, verbose=verbose)
