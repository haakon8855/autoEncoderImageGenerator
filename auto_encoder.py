"""haakon8855"""

import numpy as np
from tensorflow import keras as ks


class AutoEncoder(ks.models.Model):
    """
    Auto-encoder class for encoding images as low-dimensional representations.
    """

    def __init__(self,
                 latent_dim,
                 image_size,
                 file_name="./model_ae_std/verification_model",
                 retrain=False):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.file_name = file_name
        self.retrain = retrain

        self.encoder = ks.Sequential([
            ks.Input(shape=(image_size, image_size, 1)),
            ks.layers.Flatten(),
            ks.layers.Dense(512, activation='relu'),
            ks.layers.Dense(32, activation='relu'),
            ks.layers.Dense(latent_dim, activation='relu'),
        ])
        self.encoder.summary()
        self.decoder = ks.Sequential([
            ks.layers.Input(shape=(latent_dim)),
            ks.layers.Dense(32, activation='relu'),
            ks.layers.Dense(256, activation='relu'),
            ks.layers.Dense(512, activation='relu'),
            ks.layers.Dense(image_size**2, activation='sigmoid'),
            ks.layers.Reshape((image_size, image_size))
        ])
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

    def train(self, x_train, epochs, batch_size, shuffle, x_test):
        """
        Train the auto-encoder
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

    def call(self, x_input):
        """
        Return network output given an input x_input.
        """
        encoded = self.encoder(x_input)
        decoded = self.decoder(encoded)
        return decoded

    def generate_images(self, number_to_generate):
        """
        Generate a number of images by generating random vectors in the latent
        vector space and feeding them through the decoder.
        """
        latent_vectors = np.random.randn(number_to_generate, self.latent_dim)
        return self.decoder(latent_vectors).numpy()

    def measure_loss(self, x_test, reconstruced, check_range=200):
        """
        Measures the loss for each test sample and returns a list of losses
        corresponding to each sample in x_test on the same index.
        """
        loss = []
        for i in range(check_range):
            x_true = x_test[i, :, :, :][np.newaxis, :, :, :]
            x_pred = reconstruced[:, :, :,
                                  np.newaxis][i, :, :, :][np.newaxis, :, :, :]
            loss.append(self.evaluate(x_true, x_pred, verbose=0))
        return loss
