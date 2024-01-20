import os
import re
import time
import json
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras as K
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds



def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name],
             color='green', label='val_' + metric_name)


# Load CIFAR-10 data
training_images = np.load('cifar10_data.npz')['train_images']
training_labels = np.load('cifar10_data.npz')['train_labels']
validation_images = np.load('cifar10_data.npz')['test_images']
validation_labels = np.load('cifar10_data.npz')['test_labels']

# Preprocess input images
def preprocess_image_input(input_images):
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

# Define the feature extractor using ResNet-50
def feature_extractor(inputs):
    resnet_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    
    for layer in resnet_model.layers:
        layer.trainable = False
    
    features = resnet_model(inputs)
    return features

# Define the classifier
def classifier(inputs):
    init = tf.keras.initializers.he_normal
    l2 = tf.keras.regularizers.l2(0.0005)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu",
                              kernel_initializer=init,
                              kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(10, activation="softmax",
                              name="classification")(x)
    return x

# Define the final model
def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(224, 224))(inputs)
    
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    
    return classification_output

# Define and compile the model
# Define and compile the model
def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    
    # Use a Lambda layer to resize the input images to (224, 224)
    resized_images = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))(inputs)
    
    classification_output = final_model(resized_images)
    
    # Load the ResNet50 model with weights pre-trained on ImageNet
    resnet_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=resized_images,  # Set the input tensor to resized_images
        input_shape=(224, 224, 3),   # Set the input shape to (224, 224, 3)
        pooling=None
    )
    
    for layer in resnet_model.layers:
        layer.trainable = False
    
    resnet_features = resnet_model.outputs[0]
    
    # Flatten the ResNet features
    flattened_features = tf.keras.layers.Flatten()(resnet_features)
    
    # Add your classification layers here
    
    # Concatenate the ResNet features with your classification layers
    combined_features = tf.keras.layers.concatenate([flattened_features, classification_output])
    
    model = tf.keras.Model(inputs=inputs, outputs=combined_features)
    
    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch in [10, 20, 30]:  # Adjust the epoch numbers as needed
        lr = lr / 10.0
    return lr

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, min_delta=0.0005),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    K.callbacks.TensorBoard(log_dir="./logs")
]

# Create and train the model
model = define_compile_model()
model.summary()

EPOCHS = 100
history = model.fit(train_X, training_labels, epochs=EPOCHS,
                    validation_data=(valid_X, validation_labels), batch_size=8,
                    callbacks=callbacks)

# Save the trained model
model.save('cifar10_kaggel.h5')