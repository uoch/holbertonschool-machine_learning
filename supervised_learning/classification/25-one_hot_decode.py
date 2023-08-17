#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np


def one_hot_decode(one_hot):
    """decode one hot matrix"""
    # Y = np.argmax(one_hot, axis=0)
    classes = one_hot.shape[0]
    m = one_hot.shape[1]
    Y = []
    y = 0
    for i in range(m):
        for y in range(classes):
            if one_hot[y, i] == 1:
                Y.append(y)
                break  # Once we find the corresponding class, break the loop
    return np.array(Y)
