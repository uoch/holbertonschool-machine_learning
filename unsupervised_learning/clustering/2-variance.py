#!/usr/bin/env python3
"""clustering"""
import numpy as np


def variance(X, C):
    """Calculate intra-cluster variance for a data set"""

    distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1) ** 2
    return np.sum(np.min(distances, axis=0))
