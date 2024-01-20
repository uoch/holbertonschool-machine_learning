#!/usr/bin/env python3
""" that concatenates two matrices along a specific axis:"""
import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    """ that concatenates two matrices along a specific axis:"""
    if mat1.shape != mat2.shape:
        return None
    else:
        return np.concatenate((mat1, mat2), axis=axis)
