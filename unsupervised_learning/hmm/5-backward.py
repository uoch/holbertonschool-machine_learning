#!/usr/bin/env python3
"""markov hidden model"""

import numpy as np


def initialize_betas(X, initial, emission):
    """Initializes the backward variables in the hidden Markov model"""
    N, M = emission.shape
    T = X.shape[0]
    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    return beta


def backward(Observation, Emission, Transition, Initial):
    """backward algorithm for a hidden markov model"""
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

    beta = initialize_betas(Observation, Initial, Emission)

    for t in range(T - 2, -1, -1):
        for j in range(N):
            beta[j, t] = np.sum(
                beta[:, t + 1] * Transition[j, :] *
                Emission[:, Observation[t + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return P, beta
