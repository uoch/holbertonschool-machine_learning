#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
"""nst_class"""


class NST:
    """class NST"""
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialization"""
        tf.executing_eagerly()
        if type(style_image) is not np.ndarray or style_image.ndim != 3\
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or content_image.ndim != 3\
                or content_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not int and type(alpha) is not float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not int and type(beta) is not float) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are
          between 0 and 1 and its largest side is 512 pixels
          """
        if type(image) is not np.ndarray or\
                image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        maxd = max(h, w)
        new_h = round(h * (512/maxd))
        new_w = round(w * (512/maxd))
        image = tf.expand_dims(image, axis=0)
        image /= 255
        resized = tf.image.resize(image, size=(new_h, new_w), method="bicubic")
        return tf.clip_by_value(resized, 0.0, 1.0)
