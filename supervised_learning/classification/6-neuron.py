#!/usr/bin/env python3
"""single neuron performing binary classification"""
import numpy as np


class Neuron:
    """single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """forward propagation"""
        z = np.matmul(self.__W, X) + self.b
        # sigmoid activation function 1/(1+np.exp(-z))
        self.__A = 1/(1+np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """function cost for logistic regression"""
        loss_function = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
                          )  # -(ylog(A) + (1-y)log(1.0000001-A)))
        cost_function = 1 / Y.shape[1] * np.sum(loss_function)
        return cost_function

    def evaluate(self, X, Y):
        """evaluate the neuron"""
        a = self.forward_prop(X)
        aa = np.where(a >= 0.5, 1, 0)
        cost = self.cost(Y, a)
        return aa, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """descent function for gradient descent"""
        one_by_m = 1/X.shape[1]
        dZ = A - Y
        dW = one_by_m*(np.matmul(dZ, X.T))
        self.__W -= alpha*dW
        db = one_by_m*np.sum(dZ)
        self.__b -= alpha*db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)
        return self.evaluate(X, Y)
