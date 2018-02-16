from keras.models import Model, Input
from keras.layers import Dense, Reshape, Activation, Flatten, Concatenate, Embedding, LeakyReLU, Conv2DTranspose, UpSampling2D, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D

from layer_utils import resblock
from ac_gan import AC_GAN
from dataset import ArrayDataset
from cmd import parser_with_default_args
from train import Trainer

from conditional_layers import ConditionalInstanceNormalization, ConditionalConv2D
from keras_contrib.layers import InstanceNormalization
import keras.backend as K
import numpy as np
from sklearn.utils import shuffle


def make_generator_concat():
    x = Input(batch_shape=(64, 128, ))
    cls = Input(batch_shape=(64, 1, ), dtype='int32')

    y = Lambda(lambda c: K.cast(cls, dtype='float32'))(cls)
    y = Concatenate(axis=-1)([y, x])
    y = Dense(1024)(y)
    y = LeakyReLU()(y)
    y = Dense(128 * 7 * 7)(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)
    y = Reshape((7, 7, 128))(y)

    y = Conv2DTranspose(128, (5, 5), strides=2, padding='same')(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2D(64, (5, 5), padding='same')(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2DTranspose(64, (5, 5), strides=2, padding='same')(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2D(1, (5, 5), padding='same', activation='tanh')(y)

    return Model(inputs=[x, cls], outputs=[y])

def make_generator_ci():
    x = Input(batch_shape=(64, 128, ))
    cls = Input(batch_shape=(64, 1, ), dtype='int32')

    y = Dense(1024)(x)
    y = LeakyReLU()(y)
    y = Dense(128 * 7 * 7)(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)
    y = Reshape((7, 7, 128))(y)

    conditional_instance_norm = lambda axis: (lambda inp: ConditionalInstanceNormalization(number_of_classes=10, axis=axis)([inp, cls]))

    y = Conv2DTranspose(128, (5, 5), strides=2, padding='same')(y)
    y = conditional_instance_norm(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2D(64, (5, 5), padding='same')(y)
    y = conditional_instance_norm(axis=-1)(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(64, (5, 5), strides=2, padding='same')(y)
    y = conditional_instance_norm(axis=-1)(y)
    y = LeakyReLU()(y)
    y = Conv2D(1, (5, 5), padding='same', activation='tanh')(y)

    return Model(inputs=[x, cls], outputs=[y])

def make_separated_generator():
    x = Input(batch_shape=(64, 128, ))
    cls = Input(batch_shape=(64, 1, ), dtype='int32')

    y = Dense(1024)(x)
    y = LeakyReLU()(y)
    y = Dense(128 * 7 * 7)(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)
    y = Reshape((7, 7, 128))(y)

    y = ConditionalConv2D(128, (5, 5), number_of_classes=10, padding='same')([y, cls])
    y = UpSampling2D()(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2D(64, (5, 5), padding='same')(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = ConditionalConv2D(64, (5, 5), padding='same', number_of_classes=10)([y, cls])
    y = UpSampling2D()(y)
    y = BatchNormalization(axis=-1)(y)
    y = LeakyReLU()(y)

    y = Conv2D(1, (5, 5), padding='same', activation='tanh')(y)

    return Model(inputs=[x, cls], outputs=[y])

def make_discriminator():
    x = Input(batch_shape=(64, 28, 28, 1))

    y = Conv2D(64, (5, 5), padding='same')(x)
    y = LeakyReLU()(y)
    y = Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2])(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2], padding='same')(y)
    y = LeakyReLU()(y)
    y = Flatten()(y)
    y = Dense(1024, kernel_initializer='he_normal')(y)
    y = LeakyReLU()(y)

    cls_pred = Dense(10)(y)
    y = Dense(1)(y)

    return Model(inputs=[x], outputs=[y, cls_pred])

class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(128, )):
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        X = (X.astype(np.float32) - 127.5) / 127.5
        self._Y = Y
        self._cls_prob = np.bincount(Y) / float(Y.shape[0])
        super(MNISTDataset, self).__init__(X, batch_size, noise_size)

    def number_of_batches_per_validation(self):
        return 10

    def next_generator_sample(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size),
                np.random.choice(10, size=(self._batch_size, 1), p = self._cls_prob)]

    def next_generator_sample_test(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size),
                (np.arange(self._batch_size) % 10).reshape((self._batch_size,1))]

    def _load_discriminator_data(self, index):
        return [self._X[index], np.expand_dims(self._Y[index], axis=-1)]

    def _shuffle_data(self):
        self._X, self._Y = shuffle(self._X, self._Y)

    def display(self, output_batch, input_batch=None):
        batch = output_batch[0]
        image = super(MNISTDataset, self).display(batch)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image


def main():
    generator = make_separated_generator()
    discriminator = make_discriminator()

    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size)

    gan = AC_GAN(generator=generator, discriminator=discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()

if __name__ == "__main__":
    main()
