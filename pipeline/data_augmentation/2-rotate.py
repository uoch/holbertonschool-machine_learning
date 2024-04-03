#!/usr/bin/env python3
"""rotate_image"""
from tensorflow.image import rot90


def rotate_image(image):
    """rot90"""
    return rot90(image)