#!/usr/bin/env python3
"""that concatenates two matrices along a specific axis"""


def np_cat(mat1, mat2, axis=0):
    """that concatenates two matrices along a specific axis"""
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis)
