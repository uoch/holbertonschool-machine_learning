#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def add_arrays(arr1, arr2):
    """returns sum of two matrices"""
    if len(arr1) != len(arr2):
        return None
    else:
        result = [arr1[i] + arr2[i] for i in range(len(arr1))]
        return result
