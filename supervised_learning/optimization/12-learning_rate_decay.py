#!/usr/bin/env python3
"""Normalization Constants"""""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    alpha is the original learning rate
    decay_rate is the weight used to determi the rate at which alpha will decay
    global_step is the number of passes of gradient descent that have elapsed
    decay_step is the number of passes of gradient descent that should occur
    the learning rate decay should occur in a stepwise fashion
    """

    le = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return le
