#!/usr/bin/env python3
"""builds a neural network with the Keras library"""""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=L2))
            if len(layers) == 1:
                break
            model.add(K.layers.Dropout(1-keep_prob))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2))
            if activations[i] != 'softmax' or len(layers) == 1:
                model.add(K.layers.Dropout(1-keep_prob))
    return model
