#!/usr/bin/env python3
"""hyperparameter tuning"""
import numpy as np


class GaussianProcess:
    """ noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """gp constructor
        l = length parameter for kernel
        sigma_f = standard deviation given to output of kernel"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """kernel function
        X1: numpy.ndarray shape(m, 1)
        X2: numpy.ndarray shape(n, 1)
        k = sigma_f^2 * exp(-0.5 / l^2 * (x1 - x2)^2)
        Returns: covariance kernel matrix shape(m, n)
        """
        # (x1-x2)^2 = x1^2 + x2^2 - 2*x1*x2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        k = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
        return k
