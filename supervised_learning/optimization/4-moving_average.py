#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    v = 0
    mv_avg = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        mv_avg.append(v / (1 - beta ** (i + 1)))
    return mv_avg
