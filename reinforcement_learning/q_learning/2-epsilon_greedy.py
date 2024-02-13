#!/usr/bin/env python3
"""Q-learning module"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state,:])
    return action