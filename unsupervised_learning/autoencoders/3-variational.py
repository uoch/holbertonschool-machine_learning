#!/usr/bin/env python3
"""autoencoder"""
import tensorflow.keras as K


def create_model(input_dim, layers, latent, d=True):
    """ create encoder/decoder model"""
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
    """ get mean and log variance from latent layer"""
    mean = K.layers.Dense(latent_dims)(output)
    log_var = K.layers.Dense(latent_dims)(output)
    return mean, log_var


def sampling(args):
    """sample from normal distribution input to decoder"""
    mean, log_var = args
    epsilon = K.backend.random_normal(
        shape=(K.backend.shape(mean)[0], K.backend.int_shape(mean)[1]))
    return mean + K.backend.exp(log_var / 2) * epsilon


def vae_loss(input_img, output, mean, log_stddev):
    """vector autoencoder loss function"""
    reconstruction_loss = K.backend.sum(K.backend.binary_crossentropy(input_img, output), axis=1
            )

    # Compute the KL loss
    kl_loss = -0.5 * K.backend.sum(1 + log_stddev - K.backend.square(
        mean) - K.backend.square(K.backend.exp(log_stddev)), axis=-1)

    # Return the average loss over all images in the batch
    total_loss = K.backend.mean(reconstruction_loss + kl_loss)
    return total_loss


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    # Encoder model
    inputs, latent_output = create_model(
        input_dims, hidden_layers, latent_dims, d=False)
    mean, log_var = gaussian_mean_and_log_variance(latent_output, latent_dims)

    # Lambda layer for sampling operation
    sampling_layer = K.layers.Lambda(sampling)([mean, log_var])

    encoder_model = K.Model(inputs, [mean, log_var, sampling_layer])

    # Decoder model
    inputs_dec, decoder_output = create_model(
        latent_dims, list(reversed(hidden_layers)), input_dims, d=True)
    decoder_model = K.Model(inputs_dec, decoder_output)

    # Combined autoencoder model
    autoencoder_input = K.Input(shape=(input_dims,))
    _, _, latent_z = encoder_model(autoencoder_input)
    reconstructed_output = decoder_model(latent_z)

    # Calculate loss using the custom loss function
    loss_value = vae_loss(autoencoder_input, reconstructed_output, _, _)

    # Combined model with VAE loss
    autoencoder_model = K.Model(autoencoder_input, reconstructed_output)
    autoencoder_model.add_loss(loss_value)
    autoencoder_model.compile(optimizer='adam')

    return encoder_model, decoder_model, autoencoder_model
