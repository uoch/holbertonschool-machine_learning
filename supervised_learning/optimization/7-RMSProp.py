#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var
    """
    ss = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - (alpha * (grad / (np.sqrt(ss) + epsilon)))
    return var, ss
