#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    xx = matrix.copy()
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(i + 1, cols):  # Only loop over elements above the main diagonal
            x = xx[i][j]
            xx[i][j] = xx[j][i]
            xx[j][i] = x
    return xx