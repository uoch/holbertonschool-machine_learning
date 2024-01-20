#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward propagation of the given parameters"""
    for i in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[i], activations[i])
    return x
