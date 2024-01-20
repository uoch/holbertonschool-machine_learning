#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def cost(P, Q):
    """calculates the cost of the t-SNE transformation"""
    ep = 1e-12
    P = np.maximum(P, ep)
    Q = np.maximum(Q, ep)
    C = np.sum(P * np.log(P / Q))
    return C
