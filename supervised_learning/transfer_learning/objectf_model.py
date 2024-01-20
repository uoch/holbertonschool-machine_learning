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
    X_t = data['test_images']
    Y_t = data['test_labels']

    inter_input = K.load_model('intermedier_input.h5')
    x3 = K.layers.Flatten()(inter_input.output)
    x4 = K.layers.BatchNormalization()(x3)
    l2 = K.regularizers.l2(0.001)  # Define l2 regularization
    x5 = K.layers.Dense(128, activation='relu', kernel_regularizer=l2)(x4)
    x6 = K.layers.Dropout(0.2)(x5)
    x7 = K.layers.BatchNormalization()(x6)
    x8 = K.layers.Dense(10, activation='softmax', kernel_regularizer=l2)(x7)
    model = K.Model(inputs=inter_input, outputs=x8)  # Specify inputs and outputs

    callbacks = []

    callbacks.append(K.callbacks.LearningRateScheduler(scheduler, verbose=1))
    callbacks.append(K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, min_delta=0.0005))

    # Define the optimizer with the custom learning rate
    lr_schedule = K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
    op = K.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=op, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        X_t, Y_t, validation_split=0.2, batch_size=64, epochs=100,
        verbose=1, shuffle=True, callbacks=[callbacks, tensorboard_callback]
    )
    if max(model.history.history['val_accuracy']) >= 0.87:
        model.save('cifar10.h5')
        print("Model saved.")
