#!/usr/bin/env python3
"""random_shear"""
import tensorflow as tf


def shear_image(image, intensity):
    """Shear image"""
    # Apply shear transformation
    sheared_image = tf.keras.preprocessing.image.random_shear(image, intensity)
    return sheared_image
