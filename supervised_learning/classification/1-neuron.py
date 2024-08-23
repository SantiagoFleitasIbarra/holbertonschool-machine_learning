#!/usr/bin/env python3
"""0-neuron"""

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
