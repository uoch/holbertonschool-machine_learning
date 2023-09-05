#!/usr/bin/env python3
"""regularization"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """regularization with tensorflow"""
    return cost + tf.losses.get_regularization_losses()
