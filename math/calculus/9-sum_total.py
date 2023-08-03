#!/usr/bin/env python3
"""hat calculates \sum_{i=1}^{n} i^2:"""


def summation_i_squared(n):
    """square"""
    x = 0
    for i in range(1, n+1):
        x += i**2
    return x
