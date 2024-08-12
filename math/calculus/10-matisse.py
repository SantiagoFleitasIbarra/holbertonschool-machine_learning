#!/usr/bin/env python3
"""10-matisse"""


def poly_derivative(poly):
    """Define poly_derivative function"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for i, coef in enumerate(poly):
        if i > 0:
            derivative.append(i * coef)

    if not derivative:
        return [0]

    return derivative
