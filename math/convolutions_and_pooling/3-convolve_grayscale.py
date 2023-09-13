#!/usr/bin/env python3
"""padding"""
import numpy as np


def adjust_kernel(kernel):
    """adjust_kernel"""
    kh, kw = kernel.shape
    if kh % 2 == 0:
        ph = int(kh / 2)
    else:
        ph = int((kh - 1) / 2)
    if kw % 2 == 0:
        pw = int(kw / 2)
    else:
        pw = int((kw - 1) / 2)
    return ph, pw


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """cnvolve_grayscale_same"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    if padding == 'same':
        p_h, p_w = adjust_kernel(kernel)
        padded_images = np.pad(
            images, ((0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant', constant_values=0)

    elif padding == 'valid':
        p_h, p_w = (0, 0)
        padded_images = images

    if type(padding) == tuple:
        p_h, p_w = padding
        padded_images = np.pad(
            images, ((0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant', constant_values=0)
    if padding == 'same':
        output_h = h
        output_w = w
    else:
        output_h = h + 2 * p_h - kh + 1
        output_w = w + 2 * p_w - kw + 1
    output = np.zeros((m, output_h, output_w))
    i_step, j_step = stride
    for i in range(0, output_h, i_step):
        for j in range(0, output_w, j_step):
            zoom_in = padded_images[:, i:i+kh, j:j+kw]
            product = kernel * zoom_in
            pixel = np.sum(product, axis=(1, 2))
            output[:, i, j] = pixel
    return output
