#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    alpha is the original learning rate
    decay_rate is the wei used to determine the rate at which alpha will decay
    global_step is the number of passes of gradient descent that have elapsed
    the learning rate decay should occur in a stepwise fashion
    """
    alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return alpha
