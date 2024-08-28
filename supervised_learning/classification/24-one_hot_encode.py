#!/usr/bin/env python3
"""24-one_hot_encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """that converts a numeric label vector into a one-hot matrix"""
    try:
        m = Y.shape[0]
        one_hot = np.zeros((m, classes))
        one_hot[np.arange(m), Y] = 1
        return one_hot.T
    except Exception:
        return None
