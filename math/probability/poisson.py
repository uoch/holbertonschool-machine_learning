#!/usr/bin/env python3
"""represents a poisson distribution"""


class Poisson:
    """represents a poisson distribution"""
    e = 2.7182818285

    @staticmethod
    def factorial(x):
        """returns the factorial of x"""
        if x < 0:
            raise ValueError("x must be a positive value")
        if x == 0:
            return 1
        else:
            return x * Poisson.factorial(x - 1)

    @staticmethod
    def power(x, n):
        """returns the power of x"""
        if n == 0:
            return 1
        elif n > 0:
            result = 1
            for _ in range(n):
                result *= x
            return result
        elif n < 0:
            inv = 1
            for _ in range(int(-n)):
                inv *= x
            return 1/inv

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
            self.lambtha = float(s) / len(data)

    def pmf(self, k):
        """pmf function"""
        k = int(k)
        if k < 0:
            return 0

        p = ((self.lambtha ** k) * (self.e ** (-self.lambtha)))  / Poisson.factorial(k)
        return p
