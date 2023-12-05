#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def create_model(input_dims, filters, latent_dims, d=True):
    """creates a convolutional autoencoder"""
    revfilt = filters[::-1]
    # Encoder
    if not d:
        inputs = K.Input(shape=input_dims)
        output = inputs
        for i, conv in enumerate(filters):
            output = K.layers.Conv2D(
                conv, (3, 3), activation='relu', padding='same')(output)
            output = K.layers.MaxPooling2D((2, 2), padding='same')(output)
        return K.Model(inputs, output)
    # Decoder
    else:
        inputs = K.Input(shape=latent_dims)
        output = inputs
        for filter in revfilt[1:]:
            output = K.layers.Conv2D(
                filter, (3, 3),
                activation='relu',
                strides=(1, 1),
                padding='same')(output)
            output = K.layers.UpSampling2D((2, 2))(output)
        output = K.layers.Conv2D(
            filters=filters[0],
            kernel_size=(3, 3),
            activation='relu',
            padding='valid')(output)
        output = K.layers.UpSampling2D((2, 2))(output)
        output = K.layers.Conv2D(
            filters=input_dims[-1],
            kernel_size=(3, 3),
            activation='sigmoid',
            padding='same')(output)
        return K.Model(inputs, output)


def autoencoder(input_dims, filters, latent_dims):
    """Creates a vanilla model"""
    # Encoder
    encoder = create_model(input_dims, filters, latent_dims, d=False)
    # Decoder
    decoder = create_model(input_dims, filters, latent_dims, d=True)
    # Autoencoder
    inputs = K.Input(shape=input_dims)
    outputs = decoder(encoder(inputs))
    auto = K.Model(inputs, outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
