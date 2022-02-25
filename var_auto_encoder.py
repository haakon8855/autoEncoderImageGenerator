"""haakon8855"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks


class VariationalAutoEncoder(ks.models.Model):
    """
    VariationalAuto-encoder class for encoding images as low-dimensional representations.
    """

    def __init__(self,
                 latent_dim,
                 image_size,
                 file_name="./model_vae_std/verification_model",
                 retrain=False):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.file_name = file_name
        self.retrain = retrain

        self.encoder = ks.Sequential([
            ks.layers.InputLayer(input_shape=(image_size, image_size, 1)),
            ks.layers.Conv2D(32,
                             kernel_size=3,
                             strides=(2, 2),
                             activation='relu'),
            ks.layers.Conv2D(64,
                             kernel_size=3,
                             strides=(2, 2),
                             activation='relu'),
            ks.layers.Flatten(),
            # TODO: No activation!?
            ks.layers.Dense(latent_dim + latent_dim),
        ])
        self.encoder.summary()
        self.decoder = ks.Sequential([
            ks.layers.InputLayer(input_shape=(latent_dim, )),
            ks.layers.Dense(7 * 7 * 32, activation='relu'),
            ks.layers.Reshape((7, 7, 32)),
            ks.layers.Conv2DTranspose(64,
                                      kernel_size=3,
                                      strides=2,
                                      padding='same',
                                      activation='relu'),
            ks.layers.Conv2DTranspose(32,
                                      kernel_size=3,
                                      strides=2,
                                      padding='same',
                                      activation='relu'),
            # TODO: No activation!?
            ks.layers.Conv2DTranspose(1,
                                      kernel_size=3,
                                      strides=1,
                                      padding='same'),
            ks.layers.Reshape((image_size, image_size))
        ])
        self.optimizer = ks.optimizers.Adam(1e-4)
        self.decoder.summary()
        self.done_training = self.load_all_weights()

    def load_all_weights(self):
        """
        Load weights
        """
        # noinspection PyBroadException
        if self.retrain:
            return False
        try:
            self.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True
        except:  # pylint: disable=bare-except
            print(
                "Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False

        return done_training

    def set_optimizer(self, optimizer):
        """
        Set the optimizer to use
        """
        self.optimizer = optimizer

    def sample(self, epsilon=None):
        """
        Runs a latent representation through the decoder.
        """
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(epsilon, apply_sigmoid=True)

    def encode(self, x_input):
        """
        Runs an input (image) through the encoder network to produce
        a latent representation.
        """
        return tf.split(self.encoder(x_input), 2, 1)

    def decode(self, z_latent, apply_sigmoid=False):
        """
        Runs a latent representation z through the decoder to produce
        a reconstructed image.
        """
        logits = self.decoder(z_latent)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def reparameterize(self, mean, logvar):
        """
        Reparameterizes the latent representation using a stochastic
        vector epsilon.
        """
        epsilon = tf.random.normal(shape=mean.shape)
        return epsilon * tf.exp(logvar * 0.5) + mean

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        """
        TODO: Need to know wtf this method does
        """
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean)**2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x_input):
        """
        Computes the ELBO loss using Monte Carlo
        """
        mean, logvar = self.encode(x_input)
        z_latent = self.reparameterize(mean, logvar)
        x_logit = self.decode(z_latent)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                            labels=x_input)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = VariationalAutoEncoder.log_normal_pdf(z_latent, 0.0, 0.0)
        logqz_x = VariationalAutoEncoder.log_normal_pdf(z_latent, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_one_step(self, x_input):
        """
        Runs one training step and returns the resulting loss.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_input)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.trainable_variables))

    def train(self, x_train, epochs, batch_size, shuffle, validation_data):
        """
        Trains the VAE
        """
        epochs = 1  # TODO: remove
        for i, case in enumerate(x_train):
            self.train_one_step(x_train[i])
