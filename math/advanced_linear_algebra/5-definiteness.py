#!/usr/bin/env python3
"""advanced linear algebra"""
import numpy as np


def definiteness(matrix):
    """detemines the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if all(eigenvalues > 0):
        return "Positive definite"
    if all(eigenvalues >= 0):
        return "Positive semi-definite"
    if all(eigenvalues < 0):
        return "Negative definite"
    if all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
