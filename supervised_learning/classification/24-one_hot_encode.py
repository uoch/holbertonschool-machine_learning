#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np


def one_hot_encode(Y, classes):
    """One-hot encoding of the given classes"""
    result = np.eye(classes)[Y]
    return result
