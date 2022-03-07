"""haakon8855"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as ks


class VariationalAutoEncoder(ks.models.Model):
    """
    VariationalAuto-encoder class for encoding images as low-dimensional representations.
    """

    def __init__(self,
                 encoded_dim,
                 image_size,
                 file_name="./model_vae_std/verification_model",
                 retrain=False):
        super(VariationalAutoEncoder, self).__init__()
        self.encoded_dim = encoded_dim  # Dimension of latent/encoded representation
        self.image_size = image_size
        self.file_name = file_name
        self.retrain = retrain

        self.p_z = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=tf.zeros(encoded_dim), scale=1),
                                                 reinterpreted_batch_ndims=1)

        self.encoder = ks.Sequential([
            ks.layers.InputLayer(input_shape=(28, 28, 1)),
            ks.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            ks.layers.Conv2D(32,
                             3,
                             strides=2,
                             padding='same',
                             activation='relu'),
            ks.layers.Conv2D(64,
                             3,
                             strides=2,
                             padding='same',
                             activation='relu'),
            ks.layers.Conv2D(16,
                             3,
                             strides=1,
                             padding='same',
                             activation='relu'),
            ks.layers.Flatten(),
            ks.layers.Dense(self.get_mvntl_input_size(), activation=None),
            tfp.layers.MultivariateNormalTriL(
                self.encoded_dim,
                activity_regularizer=tfp.layers.
                KLDivergenceRegularizer(  # Apply KL-divergence as regularizer
                    self.p_z)),
        ])
        self.encoder.summary()

        self.decoder = ks.Sequential([
            ks.layers.InputLayer(input_shape=[encoded_dim]),
            ks.layers.Reshape([1, 1, encoded_dim]),
            ks.layers.Conv2DTranspose(64,
                                      7,
                                      strides=1,
                                      padding='valid',
                                      activation='relu'),
            ks.layers.Conv2DTranspose(64,
                                      3,
                                      strides=2,
                                      padding='same',
                                      activation='relu'),
            ks.layers.Conv2DTranspose(32,
                                      3,
                                      strides=2,
                                      padding='same',
                                      activation='relu'),
            ks.layers.Conv2D(1, 5, strides=1, padding='same', activation=None),
            ks.layers.Flatten(),
            tfp.layers.IndependentBernoulli(
                (image_size, image_size, 1),
                tfp.distributions.Bernoulli.logits),
        ])
        self.decoder.summary()

        self.done_training = self.load_all_weights()

    def get_mvntl_input_size(self):
        """
        Returns the size needed for the multivariate normal tril layer.
        """
        return self.encoded_dim + self.encoded_dim * (self.encoded_dim +
                                                      1) // 2

    def load_all_weights(self):
        """
        Load weights
        """
        if self.retrain:
            return False
        try:
            self.load_weights(filepath=self.file_name)
            print("Loaded model from file")
            done_training = True
        except:  # pylint: disable=bare-except
            print("Could not read weights from file. Must retrain")
            done_training = False

        return done_training

    def call(self, x_input):
        """
        Return network output given an input x_input.
        """
        encoded = self.encoder(x_input)
        decoded = self.decoder(encoded)
        return decoded

    @staticmethod
    def loss(x_input, network_output):
        """
        Computes the elbo loss. p(x|z)
        """
        return -network_output.log_prob(x_input)

    def train(self, x_train, epochs: int, batch_size: int, shuffle: bool,
              x_test):
        """
        Trains the VAE
        """
        self.done_training = self.load_all_weights()
        if not self.done_training or self.retrain:
            self.fit(x_train,
                     x_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=shuffle,
                     validation_data=(x_test, x_test))
            self.save_weights(filepath=self.file_name)
            self.done_training = True

    def generate_images(self, number_to_generate: int):
        """
        Generate a number of images by generating random vectors in the
        latent/encoded vector space and feeding them through the decoder.
        """
        encoded_vectors = self.p_z.sample(number_to_generate)
        return self.decoder(encoded_vectors).mode()[:, :, :, 0]

    def measure_loss(self,
                     x_test,
                     check_range: int = 200,
                     samples: int = 5000):
        """
        Measures the loss for each test sample and returns a list of losses
        corresponding to each sample in x_test on the same index.
        """
        generated = self.generate_images(samples).numpy()
        prob = []
        for i in range(check_range):
            loss_i = 0
            for channel in range(x_test.shape[3]):
                x_input = np.repeat(x_test[[i], :, :, [channel]],
                                    repeats=samples,
                                    axis=0)
                loss_i_channel = tf.losses.binary_crossentropy(
                    x_input.reshape(samples, 784),
                    generated.reshape(samples, 784),
                    axis=1)
                loss_i += np.exp(np.array(loss_i_channel) * -1).mean()
            prob.append(loss_i)
        return prob
