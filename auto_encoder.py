"""haakon8855"""

import numpy as np
from tensorflow import keras as ks


class AutoEncoder(ks.models.Model):
    """
    Auto-encoder class for encoding images as low-dimensional representations.
    """

    def __init__(self,
                 encoded_size,
                 image_size,
                 file_name="./model_ae_std/verification_model",
                 retrain=False):
        super(AutoEncoder, self).__init__()
        self.encoded_size = encoded_size  # Dimension of latent/encoded representation
        self.image_size = image_size  # Size of the image along one axis
        self.file_name = file_name  # File name to load and store weights from
        self.retrain = retrain  # Whether to retrain the network

        # The encoder network consisting 7 dense layers, all with relu as the
        # activation function. The layers gradually decrease in size.
        # The last layer outputs a vector of size 'encoded_size'
        # which ensures the encoded representation has a fixed size.
        self.encoder = ks.Sequential([
            ks.Input(shape=(image_size, image_size, 1)),
            ks.layers.Flatten(),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(150, activation='relu'),
            ks.layers.Dense(encoded_size, activation='relu'),
        ])
        self.encoder.summary()
        # The decoder network also consisting of 7 dense layers. The layers
        # gradually increase in size from 'encoded_size' to the original image
        # size at 28*28=784 pixels. The output is reshaped into an image.
        self.decoder = ks.Sequential([
            ks.layers.Input(shape=(encoded_size)),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
            ks.layers.Dense(1000, activation='relu'),
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
            print("Read weights successfully from file")
            done_training = True
        except:  # pylint: disable=bare-except
            print("Could not read weights from file")
            done_training = False

        return done_training

    def train(self, x_train, epochs: int, batch_size: int, shuffle: bool,
              x_test):
        """
        Train the auto-encoder
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

    def call(self, x_input):
        """
        Return network output given an input x_input.
        """
        encoded = self.encoder(x_input)
        decoded = self.decoder(encoded)
        return decoded

    def generate_images(self, number_to_generate: int):
        """
        Generate a number of images by generating random vectors in the
        latent/encoded vector space and feeding them through the decoder.
        """
        # Create a number of random vectors from the encoded space
        encoded_vectors = np.random.randn(number_to_generate,
                                          self.encoded_size)
        # Scale vectors up by a factor of 10. Originally the randomly generated
        # encoded vectors have values in the range [0.0, 1.0], but better
        # results are achieved when sampling vectors in the range [0.0, 10.0],
        # or some other interval from 0.0 to > 1.0
        encoded_vectors *= 10
        return self.decoder(encoded_vectors).numpy()

    def measure_loss(self, x_test, check_range: int = 1000):
        """
        Measures the loss for each test sample and returns a list of losses
        corresponding to each sample in x_test on the same index.
        """
        # Compute loss by iterating over a number of images from the test set,
        # running them through the network and computing the loss between the
        # original and the reconstruction.
        loss = []
        for i in range(check_range):
            loss_i = 0
            for channel in range(x_test.shape[3]):
                x_true = x_test[[i], :, :, [channel]]
                # Calculate loss for one single image
                loss_i += self.evaluate(x_true, x_true, verbose=0)
            loss.append(loss_i)
        return loss
