#!/usr/bin/env python3
"""pooling module"""
import numpy as np


def mode_choice(mode, zoom_in, axis):
    """mode choice"""
    if mode == 'max':
        return np.max(zoom_in, axis=axis)
    elif mode == 'avg':
        return np.mean(zoom_in, axis=axis)
    else:
        return None


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """pool for images with multiple channels"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    i_step, j_step = stride

    output_h = int((h-kh)/i_step+1)
    output_w = int((w-kw)/j_step+1)
    output = np.zeros((m, output_h, output_w, c))
    for i in range(0, output_h):
        x = i*i_step  # you should use the step on the image, not the output
        for j in range(0, output_w):
            y = j*j_step  # you should use step on the image, not the output
            zoom_in = A_prev[:, x:x+kh, y:y+kw, :]
            pixel = mode_choice(mode, zoom_in, axis=(1, 2))
            output[:, i, j, :] = pixel

    return output
