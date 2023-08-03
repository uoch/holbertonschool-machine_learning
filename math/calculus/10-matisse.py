#!/usr/bin/env python3
"""that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if len(poly) == 1:
        return [0]
    """
    return [poly[i+1] - poly[i] for i in range(len(poly)-1)]
    """
    x = []
    for i in range(1, len(poly)):
        x.append(poly[i] * (i))

    return x
