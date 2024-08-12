#!/usr/bin/env python3
"""17-integrate"""


def poly_integral(poly, C=0):
    """Define poly_integral function"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        integral_value = coef / (i + 1)
        if integral_value.is_integer():
            integral_value = int(integral_value)
        integral.append(integral_value)

    return integral
