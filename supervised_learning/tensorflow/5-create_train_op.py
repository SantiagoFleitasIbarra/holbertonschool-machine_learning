#!/usr/bin/env python3
"""5-create_train_op"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """create the training operation"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
