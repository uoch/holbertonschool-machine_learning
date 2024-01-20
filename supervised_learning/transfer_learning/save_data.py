import numpy as np
from keras.datasets import cifar10
from 

# Save the data to a directory
np.savez('cifar10_data.npz', train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels)