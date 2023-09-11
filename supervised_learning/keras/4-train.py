#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """Train a model with Keras and return the History object"""

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle)
