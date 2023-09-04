#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix
    sum of each row in the confusion matrix is the total
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN) = 1 - specificity = TN/matrix.sum(axis=1)
    """
    # tp is the diagonal of the confusion matrix (true positives)
    tp = np.array([confusion[i][i] for i in range(len(confusion))])
    # fn is the sum of the rows minus the diagonal (false negatives)
    fn = np.array([np.sum(confusion[i, :]) - confusion[i][i]
                  for i in range(len(confusion))])
    # fp is the sum of the columns minus the diagonal (false positives)
    fp = np.array([np.sum(confusion[:, i]) - confusion[i][i]
                  for i in range(len(confusion))])
    # tn is the sum of all cells minus the sum of the previous
    tn = np.array([np.sum(confusion) - tp[i] - fn[i] - fp[i]
                  for i in range(len(confusion))])
    return tn/(tn+fp)
