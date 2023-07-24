#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def add_arrays(arr1, arr2):
    """returns sum of two matrix"""
    add = [[arr1[i][j]+arr2[i][j]
            for i in range(arr1[0])] for j in range(arr1)]
    return add
