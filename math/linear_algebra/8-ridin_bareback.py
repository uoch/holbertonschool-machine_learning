#!/usr/bin/env python3
"""that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None

    x = [row.copy() for row in mat1]
    y = [row.copy() for row in mat2]
    h = []

    for i in range(len(x)):
        z = []
        for j in range(len(y[0])):
            dot_product = 0
            for k in range(len(y)):
                dot_product += x[i][k] * y[k][j]
            z.append(dot_product)
        h.append(z)

    return h
