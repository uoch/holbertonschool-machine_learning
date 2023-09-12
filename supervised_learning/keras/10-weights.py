#!/usr/bin/env python3
"""One-hot encode with Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves an entire model"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ loads an entire model"""
    network.load_weights(filename)
