#!/usr/bin/env python3
"""regularization"""
import tensorflow.compat.v1 as tf
import numpy as np


def l2_reg_cost(cost):
    """regularization with tensorflow"""
    return cost + tf.losses.get_regularization_losses()
