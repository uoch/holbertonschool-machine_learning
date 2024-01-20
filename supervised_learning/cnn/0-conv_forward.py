#!/usr/bin/env python3
"""cnn forward prop"""
import numpy as np


def sig(z):
    """sigmoid function"""
    return 1/(1+np.exp(-z))


def tanh(z):
    """tanh function"""
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


def relu(z):
    """relu function"""
    return np.maximum(z, 0)


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """conv forward prop"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        # solve n+2p-f+1 = n
        p_h = int(np.ceil((sh*(h_prev-1)-h_prev+kh)/2))
        p_w = int(np.ceil((sw*(w_prev-1)-w_prev+kw)/2))
    elif padding == 'valid':
        p_h, p_w = 0, 0

    elif type(padding) == tuple:
        p_h, p_w = padding

    output_h = int((h_prev+2*p_h-kh)/sh+1)
    output_w = int((w_prev+2*p_w-kw)/sw+1)
    # np.pad should be used with a tuple of 2 tuples for each dimension
    padded_images = np.pad(
        A_prev, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
        mode='constant')
    output = np.zeros((m, output_h, output_w, c_new))
    for i in range(0, output_h):
        x = i*sh  # you should use the step on the image, not the output
        for j in range(0, output_w):
            y = j*sw  # you should use step on the image, not the output
            zoom_in = padded_images[:, x:x+kh, y:y+kw, :]
            # with multiple kernels, you should sum the results of each one
            for k in range(c_new):
                kernel = W[:, :, :, k]
                product = kernel * zoom_in
                pixel = np.sum(product, axis=(1, 2, 3)) + b[:, :, :, k]
                output[:, i, j, k] = pixel

    return activation(output)
