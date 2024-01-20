#!/usr/bin/env python3
"""advanced linear algebra"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]) and len(matrix[0]) != 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    for idx, element in enumerate(matrix[0]):
        inner = []
        for row in matrix[1:]:
            inner.append(row[:idx] + row[idx + 1:])
        sign = (-1) ** idx
        det += sign * element * determinant(inner)

    return det


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    minor_output = []

    for i in range(len(matrix)):
        inner = []
        for j in range(len(matrix[0])):
            matrix_copy = [row[:] for row in matrix]
            del matrix_copy[i]
            for row in matrix_copy:
                del row[j]
            inner.append(determinant(matrix_copy))
        minor_output.append(inner)

    return minor_output


def cofactor(matrix):
    """calculates the cofactor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    minor_output = minor(matrix)

    for i in range(len(minor_output)):
        for j in range(len(minor_output[0])):
            minor_output[i][j] *= (-1) ** (i + j)

    return minor_output
