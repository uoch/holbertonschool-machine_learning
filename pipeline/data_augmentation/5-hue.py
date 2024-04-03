#!/usr/bin/env python3
"""hue image"""
import tensorflow as tf


def change_hue(image, delta):
    """hue image"""
    return tf.image.adjust_hue(image, delta)
