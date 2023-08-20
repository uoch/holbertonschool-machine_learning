#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk


class DeepNeuralNetwork:
    """deep neural network with multiple layers """

    def __init__(self, nx, layers, activation='sig'):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        if not (activation == 'sig' or activation == 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__weights = {}  # Store weight matrices
        self.__activation = activation

        # Initialize weights and biases using He initialization
        for li, layer_size in enumerate(layers):
            if not isinstance(layer_size, int) or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")
            else:
                if li == 0:
                    self.__weights['W' + str(li+1)] = np.random.randn(
                        layers[li], nx) * np.sqrt(2/nx)
                else:
                    self.weights["W"+str(li+1)] = np.random.randn(
                        layers[li], layers[li-1]) * np.sqrt(2/layers[li-1])
                self.__weights['b'+str(li+1)] = np.zeros((layer_size, 1))

        self.__L = len(layers)  # Number of layers
        self.__cache = {}

    @property
    def weights(self):
        return self.__weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def activation(self):
        return self.__activation

    @staticmethod
    def sigmoid(z):
        a = 1.0/(1.0+np.exp(-z))
        return a

    @staticmethod
    def tanh(z):
        a = 2 * DeepNeuralNetwork.sigmoid(2 * z) - 1
        return a

    @staticmethod
    def softmax(z):
        z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
        a = z_exp / np.sum(z_exp, axis=0, keepdims=True)
        return a

    def forward_prop(self, X):
        """ forward propagate using sigmoid function"""
        self.__cache['A0'] = X

        for i in range(self.__L):
            z = np.matmul(self.weights['W' + str(i + 1)],
                          self.__cache['A' + str(i)]) + \
                self.__weights['b' + str(i + 1)]
            if i == (self.__L-1):
                self.__cache['A' + str(self.__L)
                             ] = self.softmax(z)
            else:
                if self.__activation == 'sig':
                    self.__cache['A' + str(i+1)] = self.sigmoid(z)
                if self.__activation == 'tanh':
                    self.__cache['A' + str(i+1)] = self.tanh(z)

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """function cost for logistic regression
        loss_function = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
                          )  # -(ylog(A) + (1-y)log(1.0000001-A)))
        cost_function = 1 / Y.shape[1] * np.sum(loss_function)
        return cost_function"""
        # cost for softmax
        # loss = - sum(Y * np.log(
        cost_function = -1 / Y.shape[1] * np.sum(Y * np.log(A))
        return cost_function

    def evaluate(self, X, Y):
        """evaluate the neuron"""
        a1, a2 = self.forward_prop(X)
        aa = np.where(a1 == np.amax(a1, axis=0), 1, 0)
        cost = self.cost(Y, a1)
        return aa, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        o_m = 1/Y.shape[1]

        for i in range(self.__L, 0, -1):
            if i == self.__L:
                dz = cache['A' + str(self.__L)] - Y  # last layer

            ap = cache['A' + str(i - 1)]

            dw = o_m * np.matmul(dz, ap.T)
            db = o_m * np.sum(dz, axis=1, keepdims=True)
            # Calculate dz based on the chosen activation function
            # Derivation for sig activation function: (A_prev * (1 - A_prev))
            # Derivative of tanh activation function: (1 - A_prev**2)
            if self.__activation == 'sig':
                dz = np.matmul(self.__weights['W' + str(i)].T,
                               dz) * (ap * (1 - ap))
            if self.__activation == 'tanh':
                dz = np.matmul(self.__weights['W' + str(i)].T,
                               dz) * (1 - ap**2)

            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the model using the given parameters"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose and graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 and step == iterations:
                raise ValueError("step must be positive and <= iterations")
        x = []
        y = []
        for i in range(iterations):
            la, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            k = i
            if verbose and k % step == 0:
                c = self.cost(Y, self.cache['A' + str(self.__L)])
                print("cost after {} iterations: {}".format(i, c))
                x.append(i)
                y.append(c)
            if graph and k % step == 0:
                plt.plot(x, y)
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the trained model to a file"""
        if ".pkl" not in filename:
            filename += ".pkl"
        with open(filename, 'wb')as f:
            pk.dump(self, f)

    @staticmethod
    def load(filename):
        """load the traind model from a file"""
        try:
            with open(filename, 'rb')as f:
                model = pk.load(f)
            if isinstance(model, DeepNeuralNetwork):
                return model
        except FileNotFoundError:
            return None
