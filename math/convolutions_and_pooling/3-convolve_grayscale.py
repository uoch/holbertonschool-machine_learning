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


def w_h_out(w_prev, h_prev, kh, kw, sh, sw, padding, kernel):
    """ w_h_out"""
    if padding == 'same':
        ph, pw = adjust_kernel(kernel)
        output_h = ((h_prev + 2 * ph - kh) // sh) + 1
        output_w = ((w_prev + 2 * pw - kw) // sw) + 1
    if padding == 'valid':
        output_h = ((h_prev - kh) // sh) + 1
        output_w = ((w_prev - kw) // sw) + 1
    if type(padding) == tuple:
        ph, pw = padding
        output_h = ((h_prev + 2 * ph - kh) // sh) + 1
        output_w = ((w_prev + 2 * pw - kw) // sw) + 1
    return output_h, output_w


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """cnvolve_grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    i_step, j_step = stride

    if padding == 'same':
        p_h, p_w = adjust_kernel(kernel)
        padded_images = np.pad(
            images, ((0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant', constant_values=0)
    elif padding == 'valid':
        p_h, p_w = (0, 0)
        padded_images = images

    elif type(padding) == tuple:
        p_h, p_w = padding
        padded_images = np.pad(
            images, ((0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant', constant_values=0)

    output_h, output_w = w_h_out(w, h, kh, kw, i_step, j_step, padding, kernel)
    output = np.zeros((m, output_h, output_w))
    for i in range(0, output_h):
        x = i*i_step  # you should use the step on the image, not the output
        for j in range(0, output_w):
            y = j*j_step  # you should use step on the image, not the output
            zoom_in = padded_images[:, x:x+kh, y:y+kw]
            product = kernel * zoom_in
            pixel = np.sum(product, axis=(1, 2))
            output[:, i, j] = pixel

    return output
