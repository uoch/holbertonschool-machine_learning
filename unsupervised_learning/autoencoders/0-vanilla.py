#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def create_model(input_dim, layers, latent, d=True):
    inputs = K.Input(shape=(input_dim,))
    for i, layer in enumerate(layers):
        if layer == layers[0]:
            output = K.layers.Dense(layer, activation='relu')(inputs)
        else:
            output = K.layers.Dense(layer, activation='relu')(output)
        if d and i == len(layers) - 1:
            output = K.layers.Dense(latent, activation='sigmoid')(output)
    return inputs, output


def autoencoder(input_dims, hidden_layers, latent_dims):
    hidden_layers_inv = list(reversed(hidden_layers))
    inputs, encoder = create_model(
        input_dims, hidden_layers, latent_dims, d=False)
    inputs_dec, decoder = create_model(
        latent_dims, hidden_layers_inv, input_dims, d=True)
    latent = K.layers.Dense(latent_dims, activation='relu')(encoder)
    encoder_model = K.Model(inputs, latent)
    decoder_model = K.Model(inputs_dec, decoder)
    autoencoder_model = K.Model(inputs, decoder_model(encoder_model(inputs)))

    # Compile the model
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
