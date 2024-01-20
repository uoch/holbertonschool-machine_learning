#!/usr/bin/env python3
"""error analysis"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    return np.matmul(labels.T, logits)
