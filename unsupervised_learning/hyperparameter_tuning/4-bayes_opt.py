#!/usr/bin/env python3
"""hyperparameter tuning"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization"""

    def __init__(self, f, X_init, Y_init,
                 bounds,
                 ac_samples, l=1,
                 sigma_f=1,
                 xsi=0.01,
                 minimize=True):
        """Bayesian optimization constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        step = (bounds[1] - bounds[0]) / (ac_samples - 1)
        # self.X_s = np.linspace(bo[0],bo[1],ac_samples).reshape((-1, 1))
        self.X_s = np.array(
            [bounds[0] + i * step for i in range(ac_samples)])[:, np.newaxis]

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_s = np.min(self.gp.Y)
            imp = Y_s - mu - self.xsi
        else:
            Y_s = np.max(self.gp.Y)
            imp = mu - Y_s - self.xsi
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
