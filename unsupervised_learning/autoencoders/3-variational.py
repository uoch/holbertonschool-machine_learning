#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K
import tensorflow.keras as K


def create_model(input_dim, layers, latent, d=True):
    """Basic encoder model"""
    inputs = K.Input(shape=(input_dim,))
    for i, layer in enumerate(layers):
        if layer == layers[0]:
            output = K.layers.Dense(layer, activation='relu')(inputs)
        else:
            output = K.layers.Dense(layer, activation='relu')(output)
        if d and i == len(layers) - 1:
            output = K.layers.Dense(latent, activation='sigmoid')(output)
    if d:
        return inputs, output
    else:
        return inputs, output


def gaussian_mean_and_log_variance(output, latent_dims):
    """Returns the mean and log variance"""
    mean = K.layers.Dense(latent_dims)(output)
    log_var = K.layers.Dense(latent_dims)(output)
    return mean, log_var


def sampling(mean, log_var):
    """Samples from the standard normal distribution"""
    epsilon = K.backend.random_normal(shape=K.backend.shape(mean))
    return mean + K.backend.exp(log_var / 2) * epsilon


def loss(inputs, latent_output, latent_dims):
    """Loss function"""
    mu, log_var = gaussian_mean_and_log_variance(latent_output, latent_dims)
    mse = K.losses.binary_crossentropy(inputs, latent_output)
    kl = -0.5 * K.backend.sum(1 + log_var -
                              K.backend.square(mu) - K.backend.exp(log_var), axis=-1)
    loss = K.backend.mean(mse + kl)
    return loss


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a Variational Autoencoder model"""

    # Encoder model
    inputs, latent_output = create_model(
        input_dims, hidden_layers, latent_dims, d=False)
    mean, log_var = gaussian_mean_and_log_variance(latent_output, latent_dims)
    z = sampling(mean, log_var)
    encoder_model = K.Model(inputs, [mean, log_var, z])

    # Decoder model
    inputs_dec, decoder_output = create_model(
        latent_dims, list(reversed(hidden_layers)), input_dims, d=True)
    decoder_model = K.Model(inputs_dec, decoder_output)

    # Combined autoencoder model
    autoencoder_input = K.Input(shape=(input_dims,))
    _, _, latent_z = encoder_model(autoencoder_input)
    reconstructed_output = decoder_model(latent_z)

    # Loss calculation
    loss_value = loss(autoencoder_input, reconstructed_output, latent_dims)

    # Combined model with VAE loss
    autoencoder_model = K.Model(autoencoder_input, reconstructed_output)
    autoencoder_model.add_loss(loss_value)
    autoencoder_model.compile(optimizer='adam', loss=None)

    return encoder_model, decoder_model, autoencoder_model
