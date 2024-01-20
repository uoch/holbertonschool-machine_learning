#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder

np.random.seed(0)
tf.random.set_seed(0)
encoder, decoder, auto = autoencoder(784, [512, 256], 2)

# if len(auto.layers) == 3:
print(auto.layers[0].input_shape == [(None, 784)])
print(auto.layers[1] == encoder)
print(auto.layers[2] == decoder)

with open('1-test', 'w+') as f:
    x_test = mnist.load_data()[1][0]
    x_test = x_test[:256].reshape((-1, 784))
    f.write(np.format_float_scientific(auto.evaluate(x_test, x_test, verbose=False), precision=6) + '\n')
    f.write(auto.optimizer.__class__.__name__ + '\n')

with open('2-test', 'w+') as f:
    try:
        f.write(encoder.layers[0].__class__.__name__ + '\n')
        f.write(str(encoder.layers[0].input_shape) + '\n')
    except:
        f.write('FAIL\n')
    for layer in encoder.layers[1:]:
        try:
            f.write(layer.__class__.__name__ + '\n')
            if layer.__class__.__name__ == 'Dense':
                if layer.activation is not None:
                    f.write(layer.activation.__name__ + '\n')
                f.write(str(layer.input_shape) + '\n')
                f.write(str(layer.output_shape) + '\n')
            elif layer.__class__.__name__ == 'Lambda':
                f.write(str(layer.input_shape) + '\n')
                f.write(str(layer.output_shape) + '\n')
        except:
            f.write('FAIL\n')

with open('3-test', 'w+') as f:
    try:
        f.write(decoder.layers[0].__class__.__name__ + '\n')
        f.write(str(decoder.layers[0].input_shape) + '\n')
    except:
        f.write('FAIL\n')
    for layer in decoder.layers[1:]:
        try:
            f.write(layer.__class__.__name__ + '\n')
            if layer.__class__.__name__ == 'Dense' and layer.activation is not None:
                f.write(layer.activation.__name__ + '\n')
            f.write(str(layer.input_shape) + '\n')
            f.write(str(layer.output_shape) + '\n')
        except:
            f.write('FAIL\n')