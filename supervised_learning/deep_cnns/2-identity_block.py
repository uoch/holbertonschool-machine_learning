#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """identy block
    flow the this pattern:
    Conv2D(1x1) -> BatchNormalization -> ReLU
    -> Conv2D(3x3) -> BatchNormalization -> ReLU
    -> Conv2D(1x1) -> BatchNormalization -> add the input
    -> ReLU -> output
    """
    weights_int = K.initializers.he_normal()
    copy = A_prev
    F11, F3, F12 = filters
    # first layer
    layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=weights_int)(A_prev)
    # batch norm on the channels axis
    bn = K.layers.BatchNormalization(axis=3)(layer1)
    activation = K.layers.Activation('relu')(bn)
    # second layer
    layer2 = K.layers.Conv2D(filters=F3, kernel_size=(
        3, 3), padding='same',
        kernel_initializer=weights_int)(activation)
    bn2 = K.layers.BatchNormalization(axis=3)(layer2)
    act2 = K.layers.Activation('relu')(bn2)
    # third layer
    layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=weights_int)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(layer3)

    # add the copy
    fin = K.layers.Add()([bn3, copy])
    output = K.layers.Activation('relu')(fin)
    return output
