#!/usr/bin/env python3
import tensorflow.keras as K
"""builds a neural network with the Keras library"""""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library
    x track the previous layer and be the input for the next layer
    Keras.Model() is a class that takes inputs and outputs of the model"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(inputs)
            if len(layers) == 1:
                break
            x = K.layers.Dropout(1-keep_prob)(x)
        else:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(x)
            if activations[i] != 'softmax' or len(layers) == 1:
                x = K.layers.Dropout(1-keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
