#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def normalize(X, m, s):
    """normlize X matrix"""
    XX = np.subtract(X, m)
    XXX = np.divide(XX, s)
    return XXX
