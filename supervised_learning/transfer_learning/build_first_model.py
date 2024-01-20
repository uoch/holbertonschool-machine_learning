#!/usr/bin/env python3

import tensorflow.keras as K
import tensorflow as tf
import numpy as np

# Define a callback for TensorBoard
tensorboard_callback = K.callbacks.TensorBoard(log_dir="./logs")

def preprocess_data(X, Y):
    """pre-processes the data from cifar10"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def scheduler(epoch, lr):
    """scheduler function"""
    if epoch == 0:
        return lr
    else:
        return lr / (epoch * 2)

if __name__ == "__main__":
    data = np.load('cifar10_data.npz')
    X = data['train_images']
    Y = data['train_labels']

    # Define the input shape to match ResNet-50
    old_input = K.Input((32, 32, 3))  # Use the original input size

    # Resize images to (224, 224, 3) using a Lambda layer
    resized_input = K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))(old_input)

    base_model = K.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=resized_input)  # Use in
    X_p, Y_p = preprocess_data(X, Y)

    # Create a model for feature extraction
    feature_extraction_model = K.Model(inputs=old_input, outputs=base_model.output)

    # You can skip compiling the feature extraction model, as it's not used for training

    # Extract features from the training data
    train_features = feature_extraction_model.predict(X_p)

    # Save the intermediate feature extraction model
    feature_extraction_model.save('intermediate_feature_extraction.h5')
    print("Intermediate feature extraction model saved.")
