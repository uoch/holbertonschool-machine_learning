#!/usr/bin/env python3
import numpy as np
"""single neuron performing binary classification"""


class Neuron:
    """single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.random(self.nx)
        self.b = 0
        self.A = 0
