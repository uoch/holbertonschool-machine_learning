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
        x_star = np.max(self.gp.Y)
        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            mu_sample_opt = x_star - mu_sample_opt
        else:
            mu_sample_opt = np.max(self.gp.Y)
            mu_sample_opt = mu_sample_opt - x_star
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei
