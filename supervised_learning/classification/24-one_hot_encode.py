#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np


def one_hot_encode(Y, classes):
    """One-hot encoding of the given classes"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 2 or classes < Y.max():
        return None
    re = np.zeros((classes, Y.shape[0]))
    for i, y in enumerate(Y):
        re[y, i] = 1
    return re
