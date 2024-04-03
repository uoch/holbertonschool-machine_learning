#!/usr/bin/env python3
"""adjust_brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """change brightness of an image"""
    return tf.image.adjust_brightness(image, max_delta)
