#!/usr/bin/env python3
"""Implement my own transfer learning on cifar 10."""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

tensorboard_callback = K.callbacks.TensorBoard(log_dir="./logs")


def preprocess_data(X, Y):
    """Preprocess Data."""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def my_model():
    """Create model from resnet50."""
    init = K.initializers.he_normal()
    l2 = K.regularizers.l2(0.001)
    base_model = K.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=K.Input((224, 224, 3)))
    old_input = K.Input((32, 32, 3))
    data_aug = K.Sequential([
        K.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"),
        K.layers.experimental.preprocessing.RandomRotation(0.1),
        K.layers.experimental.preprocessing.RandomZoom(0.1)
    ])
    x = data_aug(old_input)
    pre = K.layers.Lambda(lambda x: tf.image.resize(x, [224, 224]))(x)
    inputs = base_model(pre)
    x1 = K.layers.Flatten()(inputs)
    x2 = K.layers.Dense(128, activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=l2,)(x1)
    x3 = K.layers.Dropout(0.2)(x2)
    x4 = K.layers.Dense(10, activation='softmax', kernel_initializer=init,
                        kernel_regularizer=l2)(x3)
    model = K.Model(old_input, x4)

    return model, base_model


def scheduler(epoch):
    """scheduler function for learning rate decay"""
    alpha = 0.001
    decay_rate = 1 / 10
    return alpha / (1 + decay_rate * epoch)


callbacks = []
callbacks.append(K.callbacks.LearningRateScheduler(scheduler, verbose=1))
callbacks.append(K.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, min_delta=0.0005))

if __name__ == "__main__":
    X = np.load('cifar10_data.npz')['train_images']
    Y = np.load('cifar10_data.npz')['train_labels']
    X_t = np.load('cifar10_data.npz')['test_images']
    Y_t = np.load('cifar10_data.npz')['test_labels']
    X_p, Y_p = preprocess_data(X, Y)
    X_tp, Y_tp = preprocess_data(X_t, Y_t)
    model, base_model = my_model()
    model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    op = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=op, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_p, Y_p, validation_data=(X_tp, Y_tp), batch_size=64, epochs=100,
              verbose=1, shuffle=True, callbacks=[callbacks, tensorboard_callback])

    if max(model.history.history['val_accuracy']) >= 0.87:
        model.save('cifar10.h5')
        print("Model saved.")
    else:
        model.save('cifar10_needmore_trainning.h5')
