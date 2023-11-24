#!/usr/bin/env python3
"""markov hidden model"""
import numpy as np


def initialize_viterbi(Observation, Emission, Transition, Initial):
    """Initializes the hmm viterbi variables"""
    N, M = Emission.shape
    T = Observation.shape[0]
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]

    return N, T, viterbi, backpointer


def viterbi(Observation, Emission, Transition, Initial):
    """dertermines the most likely sequence of hidden states for a hmm"""
    try:
        N, T, viterbi, backpointer = initialize_viterbi(
            Observation, Emission, Transition, Initial)
        for t in range(1, T):
            for j in range(N):
                viterbi[j, t] = np.max(
                    viterbi[:, t - 1] * Transition[:, j]) *\
                    Emission[j, Observation[t]]
                backpointer[j, t] = np.argmax(
                    viterbi[:, t - 1] * Transition[:, j])
        path = [0] * T
        path[T - 1] = np.argmax(viterbi[:, T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = int(backpointer[path[t + 1], t + 1])
        P = np.max(viterbi[:, T - 1:], axis=0)[0]
        return path, P
    except Exception:
        return None, None
