#!/usr/bin/env python3
"""represents a poisson distribution"""


class Exponential:
    """represents a poisson distribution"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """poisson destrubution """
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0.0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            s = sum(data)
            self.lambtha = len(data)/float(s)
