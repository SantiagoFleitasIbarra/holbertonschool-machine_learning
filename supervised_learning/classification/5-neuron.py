#!/usr/bin/env python3
"""5-neuron"""

import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
        """
        Constructor.
        nx: is the number of input features to the neuron.
        W: The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation, it
        should be initialized to 0.
        A: The activated output of the neuron (prediction). Upon instantiation,
        it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.normal(0, 1, size=(1, nx))
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
        """Calculates the forward propagation of the neuron"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W -= alpha * dw
        self.__b -= alpha * db
