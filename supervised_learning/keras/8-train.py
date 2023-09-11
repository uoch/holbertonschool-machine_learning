#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """Train a model with Keras and return the History object"""
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """scheduler function for learning rate decay"""
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(
            K.callbacks.LearningRateScheduler(scheduler, verbose=1))
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                     save_best_only=True))
    print(callbacks)
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
