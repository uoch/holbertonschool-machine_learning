#!/usr/bin/env python3
"""multivariate normal distribution"""
import numpy as np


class MultiNormal:
    """class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        data = data.T
        mean = np.mean(data, axis=0, keepdims=True)
        self.mean = mean.T
        data = data - mean
        self.cov = np.matmul(data.T, data) / (data.shape[0] - 1)
