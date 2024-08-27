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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev_layer_size = nx
        for index, layer_size in enumerate(layers, 1):
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f"W{index}"] = (np.random.randn(
                layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size))
            self.weights[f"b{index}"] = np.zeros((layer_size, 1))
            prev_layer_size = layer_size
