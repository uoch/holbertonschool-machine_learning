#!/usr/bin/env python3  

import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = np.load('cifar10_data.npz')
X = data['train_images']
Y = data['train_labels']
X_t = data['test_images']
Y_t = data['test_labels']

data_augmentation = K.Sequential([
    K.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    K.layers.experimental.preprocessing.RandomRotation(0.1),
    K.layers.experimental.preprocessing.RandomZoom(0.1)
])


