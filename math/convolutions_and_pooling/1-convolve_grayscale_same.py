#!/usr/bin/env python3
"""convolve_grayscale_valid"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """the idea is to pad the image so that the output has the same size
    the key is to check if the kernel fits in the image part to be zoomed in"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            zoom_in = images[:, i:i+kh, j:j+kw]
            _, ht, wt = zoom_in.shape
            if ht == kh and wt == kw:
                product = kernel * zoom_in
                pixel = np.sum(product, axis=(1, 2))
                output[:, i, j] = pixel
    return output
