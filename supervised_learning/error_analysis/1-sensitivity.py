#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    x = np.array([confusion[i][i]/np.sum(confusion[i])
                 for i in range(len(confusion))])
    return x
