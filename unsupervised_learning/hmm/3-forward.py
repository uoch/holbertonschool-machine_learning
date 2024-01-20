#!/usr/bin/env python3
"""markov hidden model"""

import numpy as np


def initialize_alphas(X, initial, emission):
    """Initializes the forward variables in the hidden Markov model"""
    N, M = emission.shape
    T = X.shape[0]
    alpha = np.zeros((N, T))
    alpha[:, 0] = initial.flatten() * emission[:, X[0]]
    return alpha


def forward(Observation, Emission, Transition, Initial):
    """Forward algorithm for a hidden Markov model"""
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape[0] != Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    alpha = initialize_alphas(Observation, Initial, Emission)

    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = np.sum(
                alpha[:, t - 1] * Transition[:, j]) *\
                      Emission[j, Observation[t]]

    P = np.sum(alpha[:, T - 1])
    return P, alpha
