#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """cnvolve_grayscale_same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    p_h = (kh - 1) // 2  # Vertical padding
    p_w = (kw - 1) // 2  # Horizontal padding
    output_h = h
    output_w = w
    output = np.zeros((m, output_h, output_w))
    """Padding:
    np.pad(array, pad_width, mode='constant', **kwargs)"""
    padded_images = np.pad(
        images, ((0, 0), (p_h, p_h), (p_w, p_w)), mode='constant', constant_values=0)

    for i in range(output_h):
        for j in range(output_w):
            zoom_in = padded_images[:, i:i+kh, j:j+kw]
            product = kernel * zoom_in
            pixel = np.sum(product, axis=(1, 2))
            output[:, i, j] = pixel
    return output
