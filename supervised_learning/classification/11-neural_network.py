#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np


class NeuralNetwork:
    """neural network with single hidden layer"""

    def __init__(self, nx, nodes):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))  # size =(out,in)
        # it should be initialized with 0’s.
        self.__b1 = np.zeros((self.W1.shape[0], 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))  # size =(out,in)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Forward propagation"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        # sigmoid activation function 1/(1+np.exp(-z))
        self.__A1 = 1/(1+np.exp(-z1))
        z2 = np.matmul(self.W2, self.__A1) + self.__b2
        self.__A2 = 1/(1+np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """function cost for logistic regression"""
        loss_function = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
                          )  # -(ylog(A) + (1-y)log(1.0000001-A)))
        cost_function = 1 / Y.shape[1] * np.sum(loss_function)
        return cost_function
