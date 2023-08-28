#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""

    shuffler = [i for i in range(len(X))]
    x = np.random.shuffle(shuffler)
    return X[x], Y[x]
