#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix to store the transposed elements
    xx = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            xx[j][i] = matrix[i][j]

    return xx
