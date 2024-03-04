#!/usr/bin/env python3
"""policy gradient"""
import numpy as np


def softmax(logits):
    """softmax function"""
    return np.exp(logits) / np.sum(np.exp(logits), keepdims=True)


def policy(state, weight):
    """simple policy function"""
    # flatten the state with the weight
    logits = np.dot(state, weight)
    action_probs = softmax(logits)
    return action_probs


def policy_gradient(state, weight):
    """Simple policy function"""
    # Flatten the state with the weight
    action_probs = policy(state, weight)
    action = np.random.choice(len(action_probs), p=action_probs)
    I = np.eye(len(action_probs))
    gradient = np.outer(state.T, action_probs) @ (I -
                                                  np.outer(action_probs, action_probs))

    return action, gradient
