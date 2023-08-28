#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def normalization_constants(X):
    """ calculates the normalization (standardization) constants of a matrix"""
    mean = 1/X.shape[0] * np.sum(X, axis=0)
    standard_deviation = np.sqrt(1/X.shape[0] * np.sum((X - mean)**2, axis=0))
    return mean, standard_deviation
