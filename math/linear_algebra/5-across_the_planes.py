#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def add_matrices2D(mat1, mat2):
    """returns the transpose of a 2D matrix"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        D2 = [[mat1[j][i] + mat2[j][i]
               for i in range(len(mat1[0]))] for j in range(len(mat1))]
        return D2
