#!/usr/bin/env python3
"""9-sum_total"""


def summation_i_squared(n):
    """Define summation_i_squared function"""
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return n**2 + summation_i_squared(n - 1)
