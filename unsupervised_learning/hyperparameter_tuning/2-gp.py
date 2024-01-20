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

    def predict(self, X_s):
        """predicts the mean and standard deviation of
        points in a Gaussian process"""
        k_star = self.kernel(self.X, X_s)
        k_star_star = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(self.K)
        mu_s = k_star.T@np.matmul(k_inv, self.Y)
        mu_s = mu_s.reshape(-1)
        cov_s = k_star_star - k_star.T@np.matmul(k_inv, k_star)
        # variance is on the diagonal
        cov_s = np.diag(cov_s)
        return mu_s, cov_s

    def update(self, X_new, Y_new):
        """updates a Gaussian Process"""
        self.X = np.append(self.X, X_new[:, np.newaxis], axis=0)
        self.Y = np.append(self.Y, Y_new[:, np.newaxis], axis=0)
        self.K = self.kernel(self.X, self.X)
