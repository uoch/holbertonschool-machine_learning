#!/usr/bin/env python3
"""that adds two matrices"""
import numpy as np


def add_matrices(mat1, mat2):
    """adds two matrices"""
    x = np.array(mat1)
    y = np.array(mat2)
    if x.shape != y.shape:
        return None
    else:
        return (x+y).tolist()
