#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def create_model(input_dim, layers, d=True):
    """Creates a basic encoder or decoder model"""
    inputs = K.Input(shape=(input_dim,))
    x = inputs
    for i, layer in enumerate(layers):
        x = K.layers.Dense(layer, activation='relu')(x) if i < len(
            layers) - 1 else K.layers.Dense(layer, activation='sigmoid')(x)
    return K.Model(inputs, x)


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder model"""
    # Encoder
    inputs = K.Input(shape=(input_dims,))
    encoded = create_model(input_dims, hidden_layers +
                           [latent_dims], d=False)(inputs)

    # Regularization for sparsity
    reg = K.regularizers.l1(lambtha)
    regularizer = reg(encoded)
    # Applying regularization to the encoded output
    encoded = K.layers.Lambda(lambda x: x + regularizer)(encoded)

    # Decoder
    hidden_layers_inv = list(reversed(hidden_layers))
    decoded = create_model(
        latent_dims, hidden_layers_inv + [input_dims], d=True)(encoded)

    encoder = K.Model(inputs, encoded)
    decoder = K.Model(encoded, decoded)
    autoencoder_model = K.Model(inputs, decoder(encoder(inputs)))

    # Compile the autoencoder model
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
