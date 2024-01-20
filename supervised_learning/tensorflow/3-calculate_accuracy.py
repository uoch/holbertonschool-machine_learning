#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy"""
    x = tf.cast(tf.shape(y)[0], tf.float32)
    y_pred = tf.argmax(y_pred, axis=1)
    y = tf.argmax(y, axis=1)
    true_false_array = tf.equal(y_pred, y)
    num_of_true = tf.cast(tf.reduce_sum(
        tf.cast(true_false_array, tf.int32)), tf.float32)
    accuracy = num_of_true / x
    mean_accuracy = tf.reduce_mean(accuracy, name="Mean")
    return mean_accuracy
