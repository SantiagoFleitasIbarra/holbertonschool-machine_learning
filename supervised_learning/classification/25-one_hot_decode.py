#!/usr/bin/env python3
"""25-one_hot_encode"""

import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot matrix into a vector of labels"""
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
