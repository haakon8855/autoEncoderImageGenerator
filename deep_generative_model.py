"""haakon8855"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as ks

from auto_encoder import AutoEncoder
from var_auto_encoder import VariationalAutoEncoder
from verification_net import VerificationNet
from stacked_mnist import StackedMNISTData, DataMode


class DeepGenerativeModel:
    """
    Run the deep generative model
    """

    def __init__(self):
        self.use_vae = True  # Whether to use standard AE or VAE
        self.run_anomaly_detection = True  # Whether to run anomaly detection
        self.stacked_dataset = True  # Whether to use mono or color data set
        self.encoded_dim = 5  # Size of encoded vector
        self.epochs = 45  # Number of epochs to train
        self.image_size = 28  # Size of image
        self.channels = 1  # Number of color channels
        self.batch_size = 1024
        # Anomaly detection
        self.check_for_anomalies = 1000  # Number of images in test set to check
        self.k_anomalies = 15  # Number of anomalies to plot
        # Reconstruction display
        self.number_of_reconstructions = 20  # Number of reconstructions to plot
        self.display_offset = 60  # Offset from x_test[0] to show reconstructions of
        # Generative model
        self.number_to_generate = 400  # Number of images to randomly generate
        self.generated_to_display = 20  # Number of generated images to plot
        self.learning_rate = 1e-3  # Learning rate

        self.data_set_identifier = ["mono", "color"][self.stacked_dataset]
        self.model_identifier = "ae"
        if self.use_vae:
            # Parameters specific to VAE:
            self.model_identifier = "vae"
            self.epochs = 260
            self.encoded_dim = 5
            self.check_for_anomalies = 4000
            self.anomaly_samples = 5000  # Number of generated images to compare with
            self.learning_rate = 1e-3
        if self.stacked_dataset:
            self.channels = 3

        # Set path for storing and loading trained network weights
        self.ae_weights_file_name = f"./model_{self.model_identifier}_std/verification_model"
        if self.run_anomaly_detection:
            self.ae_weights_file_name = f"./model_{self.model_identifier}_anom/verification_model"

        # Instantiate one of the auto encoders
        if self.use_vae:
            self.auto_encoder = VariationalAutoEncoder(
                self.encoded_dim,
                self.image_size,
                file_name=self.ae_weights_file_name)
        else:
            self.auto_encoder = AutoEncoder(
                self.encoded_dim,
                self.image_size,
                file_name=self.ae_weights_file_name)

        self.verification_net = None
        self.init_verification_net()  # Init/train the verification net

    def init_verification_net(self):
        """
        Train the verification net (will use stored weights as long as
        force_learn=False)
        """
        gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                               default_batch_size=2048)
        # Train the verification net.
        # If stored weights exist, load them and don't fit again.
        self.verification_net = VerificationNet(force_learn=False)
        self.verification_net.train(generator=gen, epochs=5)

    def init_auto_encoder(self):
        """
        Initializes the auto encoder, either an AE or a VAE.
        """
        if self.use_vae:
            self.auto_encoder.compile(
                optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                loss=VariationalAutoEncoder.loss,  # Custom loss function
            )
        else:
            self.auto_encoder.compile(
                optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                loss=ks.losses.BinaryCrossentropy(),
            )

    def display_reconstructions(self,
                                x_test,
                                x_pred,
                                y_test,
                                amount_to_display: int = 20,
                                offset: int = 0):
        """
        Display n original images along with their reconstruction by the AE
        """
        plt.figure(figsize=(int(amount_to_display * 0.8), 2))
        plt.gray()
        for i in range(amount_to_display):
            # Display original image
            axs = plt.subplot(2, amount_to_display, i + 1)
            # Show the image, different way depending of number of channels
            if self.stacked_dataset:
                plt.imshow(x_test[i + offset, :, :, :].astype(np.float64))
            else:
                plt.imshow(x_test[i + offset])
            plt.title(str(y_test[i + offset]).zfill(3))
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)

            # Display reconstructed image
            axs = plt.subplot(2, amount_to_display, i + 1 + amount_to_display)
            if self.stacked_dataset:
                plt.imshow(x_pred[i + offset, :, :, :].astype(np.float64))
            else:
                plt.imshow(x_pred[i + offset])
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
        plt.savefig(
            f"images/{self.model_identifier}_{self.data_set_identifier}_reconstructions"
        )  # Save image to file.png
        plt.show()  # Show same image on screen

    def display_generated(self, images, amount_to_display: int):
        """
        Display the images generated by the generative model.
        """
        # Calculate number of rows and cols to use depending of amount_to_display
        cols = int(np.ceil(np.sqrt(amount_to_display)))
        rows = int(np.ceil(amount_to_display / cols))
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(cols, rows))
        fig.suptitle("Generated images")
        plt.gray()
        i = 0
        # Iterate over the rows and columns to get each subplot in the grid.
        # For each square in the grid, use axs[row, col] to make a single plot there.
        for row in range(rows):
            for col in range(cols):
                current = axs[row, col]
                if self.stacked_dataset:
                    current.imshow(images[i, :, :, :].astype(np.float64))
                else:
                    current.imshow(images[i])
                current.get_xaxis().set_visible(False)
                current.get_yaxis().set_visible(False)
                i += 1
                if i == amount_to_display:
                    plt.savefig(
                        f"images/{self.model_identifier}_{self.data_set_identifier}_generated"
                    )
                    plt.show()
                    return

    def display_anomalies(self,
                          num_anom: int,
                          x_test,
                          loss_prob,
                          use_prob: bool = False):
        """
        Display the num_anom most anomalous image reconstructions found in
        the test set.
        """
        # Get the k most anomalous images
        if use_prob:
            # If loss_prob contains the probability of seeing x in the
            # generated images. Aka. if the model is a VAE.
            # We then find the indices of the lowest values (lowest probability).
            indices = np.argpartition(loss_prob, num_anom)[:num_anom]
        else:
            # If loss_prob contains the loss of between an original image and a
            # reconstructed one. Aka. if the model is a standard AE.
            # We then find the indices of the highest values (highest loss).
            indices = np.argpartition(loss_prob, -num_anom)[-num_anom:]
        plt.figure(figsize=(int(num_anom * 0.8), 2))
        plt.gray()
        for i in range(num_anom):
            # Display original
            axs = plt.subplot(2, num_anom, i + 1)
            if self.stacked_dataset:
                plt.imshow(x_test[indices[i], :, :, :].astype(np.float64))
            else:
                plt.imshow(x_test[indices[i]])
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)

        plt.savefig(
            f"images/{self.model_identifier}_{self.data_set_identifier}_anomalies"
        )
        plt.show()

    def detect_anomalies(self):
        """
        Runs anomaly detection.
        """
        # Fetch the complete data set containing all numbers.
        if not self.stacked_dataset:
            data_mode = DataMode.MONO_BINARY_COMPLETE
        else:
            data_mode = DataMode.COLOR_BINARY_COMPLETE
        gen_test = StackedMNISTData(mode=data_mode, default_batch_size=2048)

        # Fetch complete test data set
        x_test, _ = gen_test.get_full_data_set(training=False)

        print("Checking for anomalies")
        # Display the k_anomalies images with the most loss for
        # first 'check_for_anomalies' (e.g. 1000) samples of test set.
        if self.use_vae:
            # Fetch the probabilities of seeing each image in the test set
            prob = self.auto_encoder.measure_loss(
                x_test,
                check_range=self.check_for_anomalies,
                samples=self.anomaly_samples)
            # Display k most anomalous images (lowest probability)
            self.display_anomalies(self.k_anomalies,
                                   x_test,
                                   prob,
                                   use_prob=True)
        else:
            # Fetch the loss of each reconstructed image in the test set
            loss = self.auto_encoder.measure_loss(x_test,
                                                  self.check_for_anomalies)
            # Display k most anomalous images (highest loss)
            self.display_anomalies(self.k_anomalies, x_test, loss)

    def generate_images(self):
        """
        Generate random vectors in the latent/encoded vector-space
        and feed them through the decoder.
        """
        generated = []
        for _ in range(self.channels):
            generated.append(
                self.auto_encoder.generate_images(self.number_to_generate))
        generated = np.moveaxis(np.array(generated), 0, -1)
        self.display_generated(generated, self.generated_to_display)
        return generated

    def call(self, x_test, binary=False):
        """
        Run images through encoder and decoder
        """
        # Slight difference between passing images through AE and VAE
        if self.use_vae:
            # Output from VAE needs .mode() to function best
            encoded_imgs = self.auto_encoder.encoder(x_test)
            decoded_imgs = np.array(
                self.auto_encoder.decoder(encoded_imgs).mode())[:, :, :, 0]
            return decoded_imgs
        # Output from AE is run through haversine step-function
        # to create a crisper image.
        encoded_imgs = self.auto_encoder.encoder(x_test).numpy()
        decoded_imgs = self.auto_encoder.decoder(encoded_imgs).numpy()
        if binary:
            return np.heaviside(decoded_imgs - 0.38, 1)
        return decoded_imgs

    def run(self):
        """
        Runs the auto encoder.
        """
        # Set up data set generators/fetchers
        if not self.run_anomaly_detection and not self.stacked_dataset:
            data_mode = DataMode.MONO_BINARY_COMPLETE
        elif self.run_anomaly_detection and not self.stacked_dataset:
            data_mode = DataMode.MONO_BINARY_MISSING
        elif not self.run_anomaly_detection and self.stacked_dataset:
            data_mode = DataMode.COLOR_BINARY_COMPLETE
        else:
            data_mode = DataMode.COLOR_BINARY_MISSING

        gen_train = StackedMNISTData(mode=data_mode, default_batch_size=2048)
        gen_test = StackedMNISTData(mode=data_mode, default_batch_size=2048)

        # Fetch training and test data sets
        x_train, _ = gen_train.get_full_data_set(training=True)
        x_test, y_test = gen_test.get_full_data_set(training=False)

        self.init_auto_encoder()

        # Train the network using the training data set and predefined parameters
        print("Training network")
        self.auto_encoder.train(
            x_train[:, :, :, [0]],  # Only train on one color channel
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            x_test=x_test[:, :, :, [0]])

        if self.run_anomaly_detection:
            self.detect_anomalies()
            return

        decoded_imgs = []
        for channel in range(x_test.shape[3]):
            decoded_imgs.append(self.call(x_test[:, :, :, [channel]], True))
        decoded_imgs = np.moveaxis(np.array(decoded_imgs), 0, -1)

        print(y_test[self.display_offset:self.number_of_reconstructions +
                     self.display_offset])

        # Send reconstructed images through the verification net,
        # check coverage, predictability and accuracy for the reconstructed
        # original images.
        if self.stacked_dataset:
            reconstructed = decoded_imgs
            tolerance = 0.5  # Accuracy tolerance for stacked data set
        else:
            reconstructed = decoded_imgs[:, :, :, np.newaxis]
            tolerance = 0.8  # Accuracy tolerance for mono data set
        cov = self.verification_net.check_class_coverage(reconstructed)
        pred, acc = self.verification_net.check_predictability(
            reconstructed, y_test, tolerance=tolerance)
        # Report coverage, predictability and accuracy for the reconstructions
        print(f"Coverage: {100*cov:.2f}%")
        print(f"Predictability: {100*pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")
        self.display_reconstructions(x_test, decoded_imgs, y_test,
                                     self.number_of_reconstructions,
                                     self.display_offset)

        # Generate and show generated images
        generated_imgs = self.generate_images()
        if not self.stacked_dataset:
            # Prepare image for running through the verification net
            generated_imgs = generated_imgs[:, :, :, np.newaxis]

        gen_cov = self.verification_net.check_class_coverage(generated_imgs)
        gen_pred, _ = self.verification_net.check_predictability(
            generated_imgs)
        # Report coverage and predictability for the generated images
        print(f"Generated imgs coverage: {100*gen_cov:.2f}%")
        print(f"Generated imgs predictability: {100*gen_pred:.2f}%")


def main():
    """
    Main method for running the deep generative model.
    """
    tf.get_logger().setLevel('WARNING')
    dgm = DeepGenerativeModel()
    dgm.run()


if __name__ == "__main__":
    main()
