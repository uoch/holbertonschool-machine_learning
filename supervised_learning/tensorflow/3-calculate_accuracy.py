#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy"""
    true_false_array = tf.equal(y_pred, y)
    num_of_true = tf.reduce_sum(tf.cast(true_false_array, tf.int32))
    total = y.shape[1]
    accuracy = num_of_true/total
    mean_accuracy = tf.reduce_mean(accuracy, name="Mean")
    return mean_accuracy
