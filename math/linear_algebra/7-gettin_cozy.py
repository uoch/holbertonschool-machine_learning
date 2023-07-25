#!/usr/bin/env python3
"""concatenates two arrays in specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two arrays in specified axis"""
    x = [row.copy() for row in mat1]
    y = [row.copy() for row in mat2]

    if axis == 0:
        return x + y
    elif axis == 1:
        for i in range(len(x)):
            x[i].extend(y[i])
        return x
