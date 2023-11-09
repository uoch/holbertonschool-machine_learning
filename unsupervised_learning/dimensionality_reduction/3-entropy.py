#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def HP(Di, beta):
    """calculates Shannon entropy and P affinities relative to data point"""
    # Di = ||xi - x|| ** 2 except i = j
    Pi = np.exp(-Di * beta)/np.sum(np.exp(-Di * beta))
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
