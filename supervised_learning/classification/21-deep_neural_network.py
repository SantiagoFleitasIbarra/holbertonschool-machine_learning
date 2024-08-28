#!/usr/bin/env python3
"""16-deep_neural_network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""
    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev_layer_size = nx
        for index, layer_size in enumerate(layers, 1):
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[f"W{index}"] = (np.random.randn(
                layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size))
            self.__weights[f"b{index}"] = np.zeros((layer_size, 1))
            prev_layer_size = layer_size

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for index in range(1, self.__L + 1):
            Z = np.dot(self.__weights[f'W{index}'], self.__cache[
                f'A{index - 1}']) + self.__weights[f'b{index}']
            self.__cache[f'A{index}'] = 1 / (1 + np.exp(-Z))
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural networks predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache[f'A{L}'] - Y
        for i in range(L, 0, -1):
            A_prev = cache[f'A{i - 1}']
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * A_prev * (1 - A_prev)
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db
