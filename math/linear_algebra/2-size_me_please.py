#!/usr/bin/env python3
"""recursive method:
def matrix_shape(matrix):
    if isinstance(matrix, list):
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return []
    """
def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
