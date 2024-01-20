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
        inter_input, Y_t, v, batch_size=64, epochs=2,
        verbose=1, shuffle=True, callbacks=[callbacks, tensorboard_callback]
    )
    if max(model.history.history['val_accuracy']) >= 0.87:
        model.save('cifar10.h5')
        print("Model saved.")

