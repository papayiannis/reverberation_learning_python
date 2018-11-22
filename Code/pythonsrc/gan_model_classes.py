# Copyright 2018 Constantinos Papayiannis
#
# This file is part of Reverberation Learning Toolbox for Python.
#
# Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Reverberation Learning Toolbox for Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Reverberation Learning Toolbox for Python.  If not, see <http://www.gnu.org/licenses/>.

"""

This file provides classes which are used for constructing, training and evaluating GAN models
for reverberant acoustic environments.

Original code structure for GANs from
https://github.com/eriklindernoren/Keras-GAN
by Erik Linder-Noren @eriklindernoren

Models used for the work in:
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation using GANs for the Classification of Reverberant Rooms," (to be submitted), 2019

"""

import numpy as np
from keras import backend as K
from keras.layers import BatchNormalization, GaussianNoise
from keras.layers import Dense, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop


class GAN(object):

    def __init__(self, input_shape, net_width=256, noise_dim=20):
        """
        Constructs and trains a GAN

        Args:
            input_shape: The shape of the input
            net_width: The width of the FF layers
            noise_dim: The dimensionality of the noise input
        """
        self.latent_dim = noise_dim
        self.img_shape = input_shape
        self.net_width = net_width

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.n_discriminator = 1

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(GaussianNoise(0.1)(img))

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer,
                              metrics=['accuracy'])

    def build_generator(self):
        """
        Builds the generator network part of the GAN

        Returns:
            The Keras generator model.

        """

        model = Sequential()

        model.add(Dense(self.net_width, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.net_width))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.net_width))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """
        Builds the discriminator (critic) network

        Returns:
            The Keras model

        """
        model = Sequential()

        # model.add(Flatten())
        model.add(Dense(self.net_width, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.net_width))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, x_train, epochs, epoch_op, batch_size=64, sample_interval=500,
              nocr=False):
        """
        Trains the two models (generator and discriminator)

        Args:
            x_train: The training data from the real distribution
            epochs: The number of epochs to train for
            epoch_op: The operation to run every 'sample_interval' epochs
            batch_size: The batch size to use during training
            sample_interval: The epochs interval for the execution of 'epoch_op'
            nocr: Do not use the CR character to clear lines but start new lines

        Returns:
            Nothing

        """

        discriminator_train_k = self.n_discriminator
        disc_batch = batch_size
        cleared_ok = True
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            for _ in range(discriminator_train_k):
                idx = np.random.randint(0, x_train.shape[0], disc_batch)
                imgs = x_train[idx]

                noise = np.random.normal(0, 1, (disc_batch, self.latent_dim))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator

                # print('Discriminator : ')
                # for i in self.discriminator.layers:
                #     print('Name ' + i.name + ' Trainable ' + str(i.trainable))
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((disc_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((disc_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Plot the progress
                # printstr = str(epoch) + " [D loss: " + str(d_loss[0]) + ", acc.: " + str(
                #     100 * d_loss[1]) + "] [C loss: " + str(g_loss) + "]" + '                   \r'
                #
                # if ((epoch % 50) == 0) and not ((epoch % 100) == 0):
                #     print printstr,
            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            # print('Combined:')
            # for i in self.combined.layers:
            #     print('Name ' + i.name + ' Trainable ' + str(i.trainable))

            weights = []
            for ii in self.discriminator.layers:
                if isinstance(ii, Dense):
                    weights.append(np.concatenate(([jj.flatten() for jj in ii.get_weights()])))

            g_loss = self.combined.train_on_batch(noise, valid_y)

            for idxii, ii in enumerate(self.discriminator.layers):
                if isinstance(ii, Dense):
                    if not np.all(
                            weights[idxii] ==
                            np.concatenate(([jj.flatten() for jj in ii.get_weights()]))):
                        raise AssertionError('Layers did not freeze')

            # Plot the progress

            printstr = str(epoch) + " [D loss: " + str(d_loss[0]) + ", acc.: " + str(
                100 * d_loss[1]) + "] [C loss: " + str(g_loss[0]) + ", acc.: " + str(
                100 * g_loss[1]) + "]"

            if (epoch % 100) == 0:
                if nocr:
                    print printstr
                else:
                    print printstr + '                   \r',

            # If at save interval => save generated image samples
            cleared_ok = False or nocr
            if (epoch + 1) % sample_interval == 0:
                cleared_ok = True
                print('')
                epoch_op(self.generator, epoch)
        if not cleared_ok:
            print('')


class WGAN(GAN):

    def __init__(self, input_shape, net_width=256, noise_dim=20):
        """
        Constructs and trains a Wasserstein GAN

        Args:
            input_shape: The shape of the input
            net_width: The width of the FF layers
            noise_dim: The dimensionality of the noise input
        """
        self.latent_dim = noise_dim
        self.img_shape = input_shape
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)
        self.net_width = net_width

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        """
        Wasserstein loss functions

        Args:
            y_true: True values for y
            y_pred: Output of network

        Returns:
            The loss

        """
        return K.mean(y_true * y_pred)

    def build_critic(self):
        """
        Builds the discriminator (critic) network

        Returns:
            The Keras model

        """

        return self.build_discriminator()

    def train(self, x_train, epochs, epoch_op, batch_size=64, sample_interval=50,
              nocr=False):
        """
        Trains the two models (generator and discriminator)

        Args:
            x_train: The training data from the real distribution
            epochs: The number of epochs to train for
            epoch_op: The operation to run every 'sample_interval' epochs
            batch_size: The batch size to use during training
            sample_interval: The epochs interval for the execution of 'epoch_op'
            nocr: Do not use the CR character to clear lines but start new lines

        Returns:
            Nothing

        """

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        cleared_ok = True

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            printstr = ("%d [D loss: %f accuracy: %f] [G loss: %f]" % (
                epoch, 1 - d_loss[0], d_loss[1], 1 - g_loss[0]))

            if (epoch % 100) == 0:
                if nocr:
                    print printstr
                else:
                    print printstr + '                   \r',

            # If at save interval => save generated image samples
            cleared_ok = False
            if (epoch + 1) % sample_interval == 0:
                cleared_ok = True
                print('')
                epoch_op(self.generator, epoch)
        if not cleared_ok:
            print('')
