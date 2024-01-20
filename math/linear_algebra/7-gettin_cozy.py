#!/usr/bin/env python3
"""concatenates two arrays in specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two arrays in specified axis"""
    x = [row.copy() for row in mat1]
    y = [row.copy() for row in mat2]

    if mat1 is None or mat2 is None:
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return x + y
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(x)):
            x[i].extend(y[i])
        return x
    else:
        return None
