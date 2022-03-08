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
        self.image_size = image_size  # Size of the image along one axis
        self.file_name = file_name  # File name to load and store weights from
        self.retrain = retrain  # Whether to retrain the network

        # The latent distribution p(z)
        self.p_z = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=tf.zeros(encoded_dim), scale=1),
                                                 reinterpreted_batch_ndims=1)
        # The encoder network consisting of three convolutional layers
        # reducing the size of the image in the first two conv-layers.
        # Then a dense layer and lastly a probabilistic layer using
        # n normal distributions, where n = size of latent space.
        self.encoder = ks.Sequential([
            ks.layers.InputLayer(input_shape=(28, 28, 1)),
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
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(self.get_mvntl_input_size(), activation=None),
            tfp.layers.MultivariateNormalTriL(
                self.encoded_dim,
                activity_regularizer=tfp.layers.
                KLDivergenceRegularizer(  # Apply KL-divergence as regularizer
                    self.p_z)),
        ])
        self.encoder.summary()

        # The decoder network consisting of thee deconvolutional layers used
        # to increase the size of the image back to 28x28.
        # Strides=2 makes sure the image size is doubled after that layer
        # lastly the image is flattened and passed through an independent
        # bernoulli layer resulting in a distribution as output with the given
        # dimensions.
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
            ks.layers.Conv2D(1,
                             5,
                             strides=1,
                             padding='same',
                             activation='relu'),
            ks.layers.Flatten(),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(784, activation=None),
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
            print("Read weights successfully from file")
            done_training = True
        except:  # pylint: disable=bare-except
            print("Could not read weights from file")
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
        KL-divergence is already applied as regularizer in the last layer
        of the encoder, so this part is not included here.
        """
        return -network_output.log_prob(x_input)

    def train(self, x_train, epochs: int, batch_size: int, shuffle: bool,
              x_test):
        """
        Train the VAE
        """
        # First, check if there are stored weights available
        self.done_training = self.load_all_weights()
        # If not, then train the network
        if not self.done_training or self.retrain:
            self.fit(x_train,
                     x_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=shuffle,
                     validation_data=(x_test, x_test))
            # Save the weights to use next run
            self.save_weights(filepath=self.file_name)
            self.done_training = True

    def generate_images(self, number_to_generate: int):
        """
        Generate a number of images by generating random vectors in the
        latent/encoded vector space and feeding them through the decoder.
        """
        encoded_vectors = self.p_z.sample(number_to_generate)
        # Run the vectors through the decoder. Mode is used to get the mode
        # from the returned distribution. This produces a crisper image.
        return self.decoder(encoded_vectors).mode()[:, :, :, 0]

    def measure_loss(self,
                     x_test,
                     check_range: int = 200,
                     samples: int = 5000):
        """
        Measures the loss for each test sample and returns a list of losses
        corresponding to each sample in x_test on the same index.
        """
        # Generate a number of images to compare with
        generated = self.generate_images(samples).numpy()
        prob = []
        for i in range(check_range):
            loss_i = 0
            for channel in range(x_test.shape[3]):
                x_input = np.repeat(x_test[[i], :, :, [channel]],
                                    repeats=samples,
                                    axis=0)
                # Compute the loss between the i-th image in the test set and
                # each color channel in the generated images. This loss is the
                # binary crossentropy and the result needs to be negated and
                # e raised to the power of it before we can calculate the mean
                # of this probability.
                loss_i_channel = tf.losses.binary_crossentropy(
                    x_input.reshape(samples, 784),
                    generated.reshape(samples, 784),
                    axis=1)
                # Each probability is added to the current image
                loss_i += np.exp(np.array(loss_i_channel) * -1).mean()
            # Probability for image i in test set is appended to the list
            prob.append(loss_i)
        return prob
