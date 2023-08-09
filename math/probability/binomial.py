#!/usr/bin/env python3
""" binomial distribution"""


class Binomial:
    """Binomial distribution"""
    @staticmethod
    def factorial(x):
        """returns the factorial of x"""
        if x < 0:
            raise ValueError("x must be a positive value")
        if x == 0:
            return 1
        else:
            return x * Binomial.factorial(x - 1)

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n = int(n)
            self.p = float(p)
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
                # Estimate p from the data
            self.p = sum(data) / (len(data) * max(data))

            # Estimate n from the data using the mean and p
            # mean = np , v_square = np(1-p)
            mean_data = sum(data)/(len(data))
            sum_data = [(data[i]**2 - mean_data**2) for i in range(len(data))]
            v_square = sum(sum_data)/len(sum_data)
            self.p = (mean_data - v_square) / mean_data
            self.n = round(mean_data/self.p)
            self.p = mean_data / self.n

    def pmf(self, k):
        """probability mass function"""

        if k < 0:
            return 0 
        x = int(k)
        c = Binomial.factorial(
            self.n)/((Binomial.factorial(x)*Binomial.factorial(self.n-x)))
        pf = c * (self.p**x)*((1-self.p)**(self.n-x))
        return pf
