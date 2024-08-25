#!/usr/bin/env python3
"""0-neural_network"""

import numpy as np


class NeuralNetwork:
    """Neural Nerwork class"""
    def __init__(self, nx, nodes):
        """Constructor."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.W1 = np.random.randn(nodes, nx)
            self.b1 = np.zeros((nodes, 1))
            self.A1 = 0
            self.W2 = np.random.randn(1, nodes)
            self.b2 = 0
            self.A2 = 0