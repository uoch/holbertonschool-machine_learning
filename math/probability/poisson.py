#!/usr/bin/env python3
"""represents a poisson distribution"""


class Poisson:
    """represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
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
            self.lambtha = float(s) / len(data)
