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
