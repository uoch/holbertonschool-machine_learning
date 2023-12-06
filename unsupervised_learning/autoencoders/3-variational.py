#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K
import tensorflow as tf


class VAE_LossLayer(K.layers.Layer):
    """loss layer for variational autoencoder"""

    def __init__(self, **kwargs):
        """initialization function"""
        super(VAE_LossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """implements CVAE loss
        Args:
            inputs: list containing
                    [inputs, outputs, latent_mean, latent_log_var]"""
        inputs, outputs, latent_mean, latent_log_var = inputs

        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(
            inputs, outputs)

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + latent_log_var -
                    tf.square(latent_mean) - tf.exp(latent_log_var), axis=-1)

        # Total VAE loss
        loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(loss)
        return loss


def create_model(input_dim, layers, latent, d=True):
    """basic encoder model"""
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
    """returns the mean and log variance"""
    mean = K.layers.Dense(latent_dims)(output)
    log_var = K.layers.Dense(latent_dims)(output)
    return mean, log_var


def sampling(mean, log_var):
    """samples from the standard normal distribution"""
    epsilon = K.backend.random_normal(shape=K.backend.shape(mean))
    return mean + K.backend.exp(log_var / 2) * epsilon


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
    latent_mean, latent_log_var, latent_z = encoder_model(autoencoder_input)
    reconstructed_output = decoder_model(latent_z)
    loss_layer = VAE_LossLayer()(
        [autoencoder_input, reconstructed_output, latent_mean, latent_log_var])

    # Combined model with VAE loss
    autoencoder_model = K.Model(autoencoder_input, reconstructed_output)
    autoencoder_model.add_loss(loss_layer)
    autoencoder_model.compile(optimizer='adam', loss=None)

    return encoder_model, decoder_model, autoencoder_model
