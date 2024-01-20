
import numpy as np
import  matplotlib.pyplot as plt

data = np.load('cifar10_data.npz')
train_images, train_labels, test_images, test_labels = data['train_images'], data['train_labels'], data['test_images'], data['test_labels']
print(train_images.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
np.tile