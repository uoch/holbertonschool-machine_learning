#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def create_model(input_dim, layers, latent, lambtha, d=True):
    """basic encoder model"""
    inputs = K.Input(shape=(input_dim,))
    reg = K.regularizers.l1(lambtha)
    for i, layer in enumerate(layers):
        if layer == layers[0]:
            output = K.layers.Dense(
                layer, activation='relu', kernel_regularizer=reg)(inputs)
        else:
            output = K.layers.Dense(
                layer, activation='relu', kernel_regularizer=reg)(output)
        if d and i == len(layers) - 1:
            output = K.layers.Dense(
                latent, activation='sigmoid', kernel_regularizer=reg)(output)
    return inputs, output


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates an  vanilla model"""
    hidden_layers_inv = list(reversed(hidden_layers))
    reg1 = K.regularizers.l1(lambtha)
    inputs, encoder = create_model(
        input_dims, hidden_layers, latent_dims,lambtha, d=False)
    inputs_dec, decoder = create_model(
        latent_dims, hidden_layers_inv, input_dims,lambtha, d=True)
    latent = K.layers.Dense(latent_dims, activation='relu',
                            kernel_regularizer=reg1)(encoder)
    encoder_model = K.Model(inputs, latent)
    decoder_model = K.Model(inputs_dec, decoder)
    autoencoder_model = K.Model(inputs, decoder_model(encoder_model(inputs)))

    # Compile the model
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
