#!/usr/bin/env python3
"""0-neural_network"""

import numpy as np


class NeuralNetwork:
    """Neural Nerwork class"""
    def __init__(self, nx, nodes):
        """
        Constructor.
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.__W1 = np.random.randn(nodes, nx)
            self.__b1 = np.zeros((nodes, 1))
            self.__A1 = 0
            self.__W2 = np.random.randn(1, nodes)
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
        """Calculates the forward propagation of the neural network"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural networks predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations < 1:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")
        else:
            for i in range(iterations):
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            return self.evaluate(X, Y)
