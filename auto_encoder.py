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

    def __init__(self,
                 latent_dim,
                 image_size,
                 file_name="./model_encoder/verification_model",
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
        # self.encoder = ks.Sequential([
        #     ks.Input(shape=(image_size, image_size, 1)),
        #     ks.layers.Conv2D(8, (3, 3),
        #                      activation='relu',
        #                      padding='same',
        #                      strides=2),
        #     ks.layers.Dropout(0.25),
        #     ks.layers.Conv2D(4, (3, 3),
        #                      activation='relu',
        #                      padding='same',
        #                      strides=2),
        #     ks.layers.Dropout(0.25),
        #     ks.layers.Flatten(),
        #     ks.layers.Dense(latent_dim, activation='relu'),
        # ])
        # self.encoder.summary()
        # self.decoder = ks.Sequential([
        #     ks.Input(shape=(latent_dim)),
        #     ks.layers.Dense(image_size**2, activation='relu'),
        #     ks.layers.Reshape((image_size, image_size, 1)),
        #     ks.layers.Conv2DTranspose(4,
        #                               kernel_size=3,
        #                               activation='relu',
        #                               padding='same'),
        #     ks.layers.Dropout(0.25),
        #     ks.layers.Conv2DTranspose(8,
        #                               kernel_size=3,
        #                               activation='relu',
        #                               padding='same'),
        #     ks.layers.Dropout(0.25),
        #     ks.layers.Conv2D(1,
        #                      kernel_size=(3, 3),
        #                      activation='sigmoid',
        #                      padding='same')
        # ])
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

    def train(self, x_train, epochs, batch_size, shuffle, validation_data):
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
                     validation_data=validation_data)
            self.save_weights(filepath=self.file_name)
            self.done_training = True

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
    x_test = x_test[:, :, :, [0]]

    net = VerificationNet(force_learn=False)
    net.train(generator=gen, epochs=5)

    latent_dim = 5
    epochs = 30
    image_size = 28
    retrain = False

    auto_encoder = AutoEncoder(latent_dim, image_size, retrain=retrain)
    auto_encoder.compile(
        optimizer='adam',
        loss=ks.losses.BinaryCrossentropy(),
        # loss=ks.losses.MeanSquaredError()
        #  loss=ks.losses.categorical_crossentropy
    )
    auto_encoder.train(x_train,
                       epochs=epochs,
                       batch_size=1024,
                       shuffle=True,
                       validation_data=(x_test, x_test))

    encoded_imgs = auto_encoder.encoder(x_test).numpy()
    decoded_imgs = auto_encoder.decoder(encoded_imgs).numpy()

    print(y_test[:20])

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

    z = np.random.randn(20, latent_dim)
    generated = auto_encoder.decoder(z).numpy()
    n = 20
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display generative
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(generated[i])
        plt.title(i + 1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    # Send through verification net
    shape = decoded_imgs.shape

    cov = net.check_class_coverage(
        decoded_imgs.reshape(shape[0], shape[1], shape[2], 1))
    pred, acc = net.check_predictability(
        decoded_imgs.reshape(shape[0], shape[1], shape[2], 1), y_test)

    print(f"Coverage: {100*cov:.2f}%")
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
