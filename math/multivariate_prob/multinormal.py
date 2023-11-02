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

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2 or x.shape[1] != 1 or\
                x.shape[0] != self.cov.shape[0]:
            raise ValueError(
                'x must have the shape ({}, 1)'.format(self.cov.shape[0]))
        pi = np.pi
        p = x.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        x = x - self.mean
        pdf = 1 / np.sqrt(((2 * pi) ** p) * det) * \
            np.exp(-1 / 2 * np.matmul(np.matmul(x.T, inv), x))
        return pdf[0][0]
