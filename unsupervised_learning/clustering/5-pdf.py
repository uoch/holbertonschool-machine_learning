#!/usr/bin/env python3
"""clustering"""
import numpy as np


def pdf(X, m, S):
    """calculates the prob density function of a Gaussian distribution
    X ndarray (n, d) data points to evaluate
    m ndarray (d,) mean of distribution
    S ndarray (d, d) covariance of distribution"""

    d = X.shape[1]
    X_m = X - m
    cov_det = np.linalg.det(S)
    first_fac = 1/((np.pi*2)**(d/2) * np.sqrt(cov_det))
    # second_fac = np.einsum('...k,kl,...l->...', X_m, np.linalg.inv(S), X_m)
    second_fac = np.sum(np.matmul(X_m, np.linalg.inv(S)) * X_m, axis=1)
    pdf = first_fac * np.exp(-second_fac/2)
    pdf = np.where(pdf < 1e-300, 1e-300, pdf)
    return pdf
