#!/usr/bin/env python3
"""hat calculates"""


def summation_i_squared(n):
    """square"""
    if not isinstance(n, int) or n <= 0:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
