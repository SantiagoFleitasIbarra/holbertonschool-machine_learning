#!/usr/bin/env python3
"""0-line"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Define line"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
