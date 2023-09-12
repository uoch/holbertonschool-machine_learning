#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves model config"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """lodes model config"""
    with open(filename, 'r') as f:
        json_string = f.read()
        return K.models.model_from_json(json_string)
