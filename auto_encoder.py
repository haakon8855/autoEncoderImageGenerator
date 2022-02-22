"""Haakon8855"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as ks

from verification_net import VerificationNet
from stacked_mnist import StackedMNISTData, DataMode


class AutoEncoder(ks.models.Model):
    """
    Auto-encoder class for encoding images as low-dimensional representations.
    """

    def __init__(self, latent_dim, image_size):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = ks.Sequential([
            ks.layers.Flatten(),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = ks.Sequential([
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(500, activation='relu'),
            ks.layers.Dense(image_size**2, activation='sigmoid'),
            ks.layers.Reshape((image_size, image_size))
        ])
        # self.encoder = ks.Sequential([
        #     ks.Input(shape=(image_size, image_size, 1)),
        #     ks.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        #     # ks.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        #     ks.layers.Flatten(),
        #     ks.layers.Dense(latent_dim, activation='relu'),
        # ])
        # self.decoder = ks.Sequential([
        #     ks.layers.Dense(image_size**2, activation='relu'),
        #     ks.layers.Reshape((image_size, image_size, 1)),
        #     # ks.layers.Conv2DTranspose(8,
        #     #                           kernel_size=3,
        #     #                           activation='relu',
        #     #                           padding='same'),
        #     ks.layers.Conv2DTranspose(16,
        #                               kernel_size=3,
        #                               activation='relu',
        #                               padding='same'),
        #     ks.layers.Conv2D(1,
        #                      kernel_size=(3, 3),
        #                      activation='sigmoid',
        #                      padding='same'),
        # ])

    def call(self, x_input):
        """
        Return network output given an input x_input.
        """
        encoded = self.encoder(x_input)
        decoded = self.decoder(encoded)
        return decoded


def main():
    """
    Main function for running the auto encoder.
    """
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                           default_batch_size=2048)

    x_train, y_train = gen.get_full_data_set(training=True)
    x_test, y_test = gen.get_full_data_set(training=False)
    # "Translate": Only look at "red" channel;
    # only use the last digit. Use one-hot for labels during training
    x_train = x_train[:, :, :, [0]]
    y_train = ks.utils.to_categorical((y_train % 10).astype(np.int), 10)
    x_test = x_test[:, :, :, [0]]
    y_test = ks.utils.to_categorical((y_test % 10).astype(np.int), 10)

    net = VerificationNet(force_learn=False)
    net.train(generator=gen, epochs=5)

    latent_dim = 2
    epochs = 20
    image_size = 28

    auto_encoder = AutoEncoder(latent_dim, image_size)
    auto_encoder.compile(
        optimizer='adam',
        loss=ks.losses.BinaryCrossentropy(),
        # loss=ks.losses.MeanSquaredError()
        #  loss=ks.losses.categorical_crossentropy
    )
    auto_encoder.fit(x_train,
                     x_train,
                     epochs=epochs,
                     batch_size=1024,
                     shuffle=True,
                     validation_data=(x_test, x_test))

    encoded_imgs = auto_encoder.encoder(x_test).numpy()
    decoded_imgs = auto_encoder.decoder(encoded_imgs).numpy()

    n = 20
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title(i + 1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title(i + 1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    main()
