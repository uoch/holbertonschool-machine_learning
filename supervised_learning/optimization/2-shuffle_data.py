#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""

    ind = np.random.permutation(len(X))
    # ind is a list of random numbers from 0 to len(X)
    return X[ind], Y[ind]
