#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ Train a model with Keras and return the History object
    callbacks is the list of callbacks to be called during training
    which includes early stopping during training"""

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
