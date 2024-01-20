#!/usr/bin/env python3
"""normal distribution"""


class Normal:
    """normal"""
    π = 3.1415926536
    e = 2.7182818285

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

    def z_score(self, x):
        """z_score"""
        z = x - self.mean
        zs = z/self.stddev
        return zs

    def x_value(self, z):
        """x_value"""
        zs = z*self.stddev
        x = self.mean + zs
        return x

    def pdf(self, x):
        """probability density function"""
        le = 1/(self.stddev*((self.π*2)**0.5))
        r = self.e**((-(x-self.mean)**2)/((self.stddev**2)*2))
        pf = le*r
        return pf

    def erf(self, x):
        """erreur function"""
        le = 2/((self.π)**0.5)
        r = x - ((x**3)/3) + ((x**5)/10)-((x**7)/42)+((x**9)/216)
        return le*r

    def cdf(self, x):
        """cumulative density function"""
        X = (x - self.mean)/((2**0.5)*self.stddev)
        cf = (1/2)*(1 + self.erf(X))
        return cf
