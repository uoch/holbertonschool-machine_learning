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

    def evaluate(self, X, Y):
        """evaluate the neuron"""
        a1, a2 = self.forward_prop(X)
        aa = np.where(a2 >= 0.5, 1, 0)
        cost = self.cost(Y, a2)
        return aa, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradien descent"""
        one_by_m = 1/X.shape[1]
        dZ2 = A2 - Y

        dW2 = one_by_m*(np.matmul(dZ2, A1.T))

        db2 = one_by_m*np.sum(dZ2, axis=1, keepdims=True)

        # dZ1 = (W2.T * dZ2) * (A1 * (1 - A1))
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1-A1))
        dW1 = one_by_m*np.matmul(dZ1, X.T)  # dW1 = (1/m) * (dZ1 * X.T)

        # db1 = (1/m) * sum(dZ1)
        db1 = one_by_m * np.sum(dZ1, axis=1, keepdims=True)
        self.__W2 -= alpha*dW2
        self.__b2 -= alpha*db2
        self.__W1 -= alpha*dW1
        self.__b1 -= alpha*db1
        return self.__W1, self.__b1, self.__W2, self.__b2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the model using the given parameters"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
