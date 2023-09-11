#!/usr/bin/env python3
"""builds a neural network with the Keras library"""""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    dropout = K.layers.Dropout(1-keep_prob)
    layer = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=L2, input_shape=(nx,))
    model.add(layer)
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=L2))
        if i != len(layers) - 1:
            model.add(dropout)
    return model
