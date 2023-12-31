#!/usr/bin/env python3
"""regularization"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return (count >= patience, count)
