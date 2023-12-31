#!/usr/bin/env python3
"""hot encode with keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, classes)
