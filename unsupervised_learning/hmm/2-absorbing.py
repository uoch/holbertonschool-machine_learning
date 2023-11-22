#!/usr/bin/env python3
"""markov hidden model"""
import numpy as np


def absorbing(P):
    """Determines if the given matrix represents an absorbing Markov chain"""
    if not isinstance(P, np.ndarray) or P.ndim != 2 \
            or P.shape[0] != P.shape[1]:
        return False

    diag = np.diagonal(P)
    num_states = P.shape[0]

    if not np.any(diag == 1):
        return False

    absorbing_states = np.where(diag == 1)[0]

    # x is the a list of absorbing states
    x = absorbing_states.tolist()
    while True:
        # track is a copy of x that will be used to check if x has changed
        track = x.copy()

        for i in range(num_states):
            if i not in x:
                if any(P[i, j] != 0 for j in x):
                    x.append(i)

        if len(track) == len(x):
            break

    return len(x) == num_states
