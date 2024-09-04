#!/usr/bin/env python3
"""2-forward_prop"""

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward_prop"""
    output = x
    for i in range(len(layer_sizes)):
        activation = activations[i] if activations[i] is not None else None
        output = create_layer(output, layer_sizes[i], activations)
    return output
