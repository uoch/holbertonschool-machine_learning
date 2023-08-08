#!/usr/bin/env python3
"""normal distribution"""


class Normal:
    """normal"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
            if stddev <= 0.0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            num = [(data[i]-self.mean)**2 for i in range(len(data))]
            den = len(data)
            self.stddev = (sum(num)/den)**0.5
