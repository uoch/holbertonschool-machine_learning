#!/usr/bin/env python3
"""backward prop over a conv layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """backward prop over a conv layer of a neural network
    dz: np.ndarray (m, h_new, w_new, c_new) containing the partial derivatives
    of the unactivated output of the convolutional layer
    A_prev: np.ndarray (m, h_prev, w_prev, c_prev) output of the previous layer
    W: np.ndarray (kh, kw, c_prev, c_new) kernels for the convolution
    b: np.ndarray (1, 1, 1, c_new) biases applied to the convolution
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((sh*(h_prev-1)-h_prev+kh)/2))
        pw = int(np.ceil((sw*(w_prev-1)-w_prev+kw)/2))
    elif padding == 'valid':
        ph, pw = 0, 0
    elif type(padding) == tuple:
        ph, pw = padding
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dw = np.zeros(W.shape)
    dA = np.zeros(A_prev_pad.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for m in range(m):
        for i in range(h_new):
            # we should loop over gradient's height and width
            for j in range(w_new):
                x = i * sh  # use the step on the image, not the output
                y = j * sw
                for k in range(c_new):
                    # Get a slice from A_prev_pad
                    zoom = A_prev_pad[m, x:x+kh, y:y+kw, :]
                    # Compute the gradient for the current neuron in layer l
                    dzK = dZ[m, i, j, k]
                    # Get the kernel (filter) for the current neuron in layer l
                    kernel = W[:, :, :, k]
                    # Accumulate the gradient contribution to dA for layer l-1
                    dA[m, x:x+kh, y:y+kw, :] += dzK * kernel
                    # Accumulate the gradient contribution to dw for layer l
                    dw[:, :, :, k] += zoom * dzK
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]

    return dA, dw, db
