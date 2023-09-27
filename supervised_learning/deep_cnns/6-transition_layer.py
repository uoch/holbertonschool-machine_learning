#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """transition layer for DenseNet
    Batch Normalization-ReLU-convolution1x1 - AveragePooling 2D stride = 2
    compression is the compression factor for the transition layer
    """
    comp = int(nb_filters*compression)
    batch1 = K.layers.BatchNormalization(axis=3)(X)
    activation1 = K.layers.Activation('relu')(batch1)
    conv1 = K.layers.Conv2D(filters=comp,
                            kernel_size=(1, 1), 
                            padding='same',
                            kernel_initializer='he_normal')(activation1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same')(conv1)
    return avg_pool, int(nb_filters*compression)
