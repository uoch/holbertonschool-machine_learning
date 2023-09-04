#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix
    f1 = 2tp / (2tp + fp + fn)
    """
    tp = np.array([confusion[i][i] for i in range(len(confusion))])
    fn = np.array([np.sum(confusion[i, :]) - confusion[i][i]
                  for i in range(len(confusion))])
    fp = np.array([np.sum(confusion[:, i]) - confusion[i][i]
                  for i in range(len(confusion))])
    f1 = 2*tp / (2*tp + fp + fn)
    return f1
