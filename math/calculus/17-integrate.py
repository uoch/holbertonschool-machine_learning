#!/usr/bin/env python3
"""that calculates the integral of a polynomial:"""


def poly_integral(poly, C=0):
    """return sum(poly[i] * (i + 1) ** C for i in range(len(poly) - 1))"""
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    x = []
    x.append(C)
    for i in range(len(poly)):
        k = poly[i] / (i+1)
        if k.is_integer():
            x.append(int(k))
        else:
            x.append(k)
    return x
