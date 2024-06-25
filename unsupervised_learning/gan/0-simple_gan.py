import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):

    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        # run the __init__ of keras.Model first.
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta_1 = .5
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    # generator of real samples of size batch_size

    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        # Training the discriminator
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get a batch of real samples
                real_sample = self.get_real_sample()
                # Get a batch of fake samples
                fake_sample = self.get_fake_sample()
                # Compute discriminator loss
                discr_loss = self.discriminator.loss(self.discriminator(
                    real_sample), self.discriminator(fake_sample))
            # Compute gradients and apply to discriminator
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_weights)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights))

        # Training the generator
        with tf.GradientTape() as tape:
            # Generate a batch of fake samples
            fake_sample = self.get_fake_sample()
            # Compute generator loss
            gen_loss = self.generator.loss(self.discriminator(fake_sample))
        # Compute gradients and apply to generator
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        # Return losses for monitoring
        return {"gen_loss": gen_loss, "discr_loss": discr_loss}
