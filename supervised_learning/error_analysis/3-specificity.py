#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix
    sum of each row in the confusion matrix is the total
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN) = 1 - specificity = TN/matrix.sum(axis=1)
    """
    x = np.array([confusion[i][i]/(np.sum(confusion[i]) - confusion[i][i])
                 for i in range(len(confusion))])
    return x
