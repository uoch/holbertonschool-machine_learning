#!/usr/bin/env python3
"""attention"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """generates positional encoding matrix"""
    matrix = np.zeros((max_seq_len, dm))
    n = 10000
    for i in range(max_seq_len):
        for j in range(dm//2):
            matrix[i, (2*j)] = np.sin(i/(n**(2*j/dm)))
            matrix[i, (2*j)+1] = np.cos(i/(n**(2*j/dm)))
    return matrix
