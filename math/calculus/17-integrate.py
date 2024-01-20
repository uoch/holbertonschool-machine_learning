#!/usr/bin/env python3
"""that calculates the integral of a polynomial:"""


def poly_integral(poly, C=0):
    """return sum(poly[i] * (i + 1) ** C for i in range(len(poly) - 1))"""
    if not isinstance(poly, list):
        return None
    x = []
    if len(poly) == 1 and isinstance(poly[0], int) and isinstance(C, int):
        if poly[0] != 0:
            poly.append(C)
            poly.reverse()
            return poly
        else:
            return [C]
    elif len(poly) > 1:
        x = []
        x.append(C)
        for i in range(len(poly)):
            k = poly[i] / (i+1)
            if k.is_integer():
                x.append(int(k))
            else:
                x.append(k)
        return x
