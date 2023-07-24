#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(i + 1, cols):  # Only loop over elements above the main diagonal
            x = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = x
