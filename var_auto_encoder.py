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
                 latent_dim,
                 image_size,
                 file_name="./model_vae_std/verification_model",
                 retrain=False):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.file_name = file_name
        self.retrain = retrain

        self.p_z = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=tf.zeros(latent_dim), scale=1),
                                                 reinterpreted_batch_ndims=1)

        self.encoder = ks.Sequential([
            ks.Input(shape=(image_size, image_size, 1)),
            ks.layers.Flatten(),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(40, activation='relu'),
            ks.layers.Dense(self.get_mvntl_input_size(), activation=None),
            tfp.layers.MultivariateNormalTriL(
                self.latent_dim,
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                    self.p_z)),
        ])
        self.encoder.summary()

        self.decoder = ks.Sequential([
            ks.layers.Input(shape=(latent_dim)),
            ks.layers.Dense(40, activation='relu'),
            ks.layers.Dense(250, activation='relu'),
            ks.layers.Dense(400, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(image_size**2),
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
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2

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
        Generate a number of images by generating random vectors in the latent
        vector space and feeding them through the decoder.
        """
        latent_vectors = self.p_z.sample(number_to_generate)
        return self.decoder(latent_vectors).mode()

    def measure_loss_by_sampling(self, x_test, check_range: int = 200):
        """
        Measures the loss for each test sample and returns a list of losses
        corresponding to each sample in x_test on the same index.
        """
        N = 5000
        check_range = 4000
        generated = self.generate_images(N).numpy()
        prob = []
        for i in range(check_range):
            x_input = np.repeat(x_test[[i], :, :, :], repeats=N, axis=0)
            loss_i = tf.losses.binary_crossentropy(x_input.reshape(N, 784),
                                                   generated.reshape(N, 784),
                                                   axis=1)
            prob.append(np.exp(np.array(loss_i) * -1).mean())
        return prob
