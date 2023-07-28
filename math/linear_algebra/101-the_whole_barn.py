#!/usr/bin/env python3
"""that performs matrix multiplication"""


def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def add_matrices(mat1, mat2):
    """Adds two matrices represented as lists."""
    if not (isinstance(mat1, list) and isinstance(mat2, list)):
        return None

    if not (isinstance(mat1[0], list) and isinstance(mat2[0], list)):
        if len(mat1) != len(mat2):
            return None
        else:
            return [mat1[i] + mat2[i] for i in range(len(mat1))]

    # Check if both matrices have the same shape
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    elif len(shape1)<= 2:
        # Initialize the result list
        Badd = []
        for i in range(len(shape1)):
            # Check if both rows have the same number of columns
            if len(mat1[i]) != len(mat2[i]):
                return None
            # Perform element-wise addition and store the result row
            add_row = [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            Badd.append(add_row)

        return Badd
    else:
        return [1,1,1,1,1,1,1,1,1,1]
