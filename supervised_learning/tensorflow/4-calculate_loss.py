#!/usr/bin/env python3
"""4-calculate_loss"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
