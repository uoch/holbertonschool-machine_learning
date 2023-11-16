#!/usr/bin/env python3
"""clustering"""
import numpy as np


def variance(X, C):
    """Calculate intra-cluster variance for a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if (not isinstance(C, np.ndarray) or len(C.shape) != 2 or
            C.shape[1] != X.shape[1]):
        return None

    distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1) ** 2
    return np.sum(np.min(distances, axis=0))
