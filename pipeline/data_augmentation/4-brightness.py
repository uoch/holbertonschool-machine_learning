#!/usr/bin/env python3
"""adjust_brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    return tf.image.adjust_brightness(image, max_delta)
