#!/usr/bin/env python3
"""error analysis"""
import numpy as np

def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    m = np.sum(confusion[0])
    x = np.array([confusion[i][i]/m for i in range(len(confusion))])
    return x