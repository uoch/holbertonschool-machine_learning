#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix
    False Positives (FP) for Class i: FP_i = sum(conf[:, i]) - conf[i][i]
    False Negatives (FN) for Class i: FN_i = sum(conf[i, :]) - conf[i][i]
    True Negatives (TN) for Class i: TN_i = sum(sum(conf)) - TP_i - FP_i - FN_i
    precision = TP / (TP + FP)
"""
    x = np.array([confusion[i][i]/(np.sum(confusion[:, i]))
                 for i in range(len(confusion))])
    return x
