#!/usr/bin/env python3
"""Get shape of array."""


def matrix_shape(matrix):
    """recursive method:
def matrix_shape(matrix):
    if isinstance(matrix, list):
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return []
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
