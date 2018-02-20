from keras.models import Model, Input
from keras.layers import Dense, Reshape, Activation, Flatten, Concatenate, Embedding, LeakyReLU, Conv2DTranspose, UpSampling2D, Lambda, BatchNormalization, Add, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D

from layer_utils import resblock, glorot_init, he_init
from gan import GAN
from ac_gan import AC_GAN
from dataset import ArrayDataset
from cmd import parser_with_default_args
from train import Trainer

from conditional_layers import ConditionalInstanceNormalization, ConditionalConv2D, cond_resblock
import keras.backend as K
import numpy as np
from sklearn.utils import shuffle

def make_generator_concat():
    x = Input((128, ))
    cls = Input((1, ), dtype='int32')

    y = Lambda(lambda c: K.cast(cls, dtype='float32'))(cls)
    y = Concatenate(axis=-1)([y, x])
    y = Dense(128 * 7 * 7, kernel_initializer=glorot_init)(y)
    y = Reshape((7, 7, 128))(y)

    y = resblock(y, (3, 3), 'UP', 128, BatchNormalization)
    y = resblock(y, (3, 3), 'UP', 128, BatchNormalization)

    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), kernel_initializer=glorot_init, use_bias=True,
                      padding='same', activation='tanh')(y)
    return Model(inputs=[x, cls], outputs=y)


def make_generator_resnet_sep():
    x = Input((128, ))
    cls = Input((1, ), dtype='int32')

    y = Dense(128 * 7 * 7, kernel_initializer=glorot_init)(x)
    y = Reshape((7, 7, 128))(y)

    y = cond_resblock(y, cls, (3, 3), 'UP', 128, number_of_classes=10)
    y = cond_resblock(y, cls, (3, 3), 'UP', 128, number_of_classes=10)

    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), kernel_initializer=glorot_init, use_bias=True,
                      padding='same', activation='tanh')(y)
    return Model(inputs=[x, cls], outputs=y)


def make_generator_resnet_ci():
    x = Input((128, ))
    cls = Input((1, ), dtype='int32')

    y = Dense(128 * 7 * 7, kernel_initializer=glorot_init)(x)
    y = Reshape((7, 7, 128))(y)

    conditional_instance_norm = lambda axis: (lambda inp: ConditionalInstanceNormalization(number_of_classes=10, axis=axis)([inp, cls]))

    y = resblock(y, (3, 3), 'UP', 128, conditional_instance_norm)
    y = resblock(y, (3, 3), 'UP', 128, conditional_instance_norm)

    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), kernel_initializer=glorot_init, use_bias=True,
                      padding='same', activation='tanh')(y)
    return Model(inputs=[x, cls], outputs=y)


def make_discriminator_resnet():
    x = Input((28, 28, 1))

    y = resblock(x, (3, 3), 'DOWN', 128, norm=None, is_first=True)
    y = resblock(y, (3, 3), 'DOWN', 128, norm=None)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)

    y = Activation('relu')(y)

    y = GlobalAveragePooling2D()(y)
    cls_out = Dense(10, use_bias=True, kernel_initializer=glorot_init)(y)
    y = Dense(1, use_bias=True, kernel_initializer=glorot_init)(y)

    return Model(inputs=x, outputs=[y, cls_out])

def make_discriminator_resnet_sep():
    x = Input((28, 28, 1))
    cls = Input((1, ), dtype='int32')

    y = cond_resblock(x, cls, (3, 3), 'DOWN', 128, number_of_classes=10, norm=None, is_first=True)
    y = cond_resblock(y, cls, (3, 3), 'DOWN', 128, number_of_classes=10, norm=None)
    y = cond_resblock(y, cls, (3, 3), 'SAME', 128, number_of_classes=10, norm=None, conv_shortcut=False)
    y = cond_resblock(y, cls, (3, 3), 'SAME', 128, number_of_classes=10, norm=None, conv_shortcut=False)

    y = Activation('relu')(y)

    y = GlobalAveragePooling2D()(y)
    y = Dense(1, use_bias=True, kernel_initializer=glorot_init)(y)

    return Model(inputs=[x, cls], outputs=[y])


def make_discriminator_projective():
    x = Input((28, 28, 1))
    cls = Input((1, ), dtype='int32')

    y = resblock(x, (3, 3), 'DOWN', 128, norm=None, is_first=True)
    y = resblock(y, (3, 3), 'DOWN', 128, norm=None)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)

    y = Activation('relu')(y)

    y = GlobalAveragePooling2D()(y)

    psi = Dense(1, kernel_initializer=glorot_init)(y)
    emb = Embedding(input_dim=10, output_dim=128)(cls)
    phi = Lambda(lambda inp: K.sum(K.squeeze(inp[0], axis=1) * inp[1], axis=1), output_shape=(1, ))([emb, y])

    y = Add()([psi, phi])

    return Model(inputs=[x, cls], outputs=[y])


def make_spectral_discriminator():
    from spectral_normalized_layers import SNConv2D, SNDense
    x = Input((28, 28, 1))

    y = resblock(x, (3, 3), 'DOWN', 128, norm=None, is_first=True, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'DOWN', 128, norm=None, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False, conv_layer=SNConv2D)

    y = Activation('relu')(y)

    y = GlobalAveragePooling2D()(y)
    cls_out = SNDense(units=10, use_bias=True, kernel_initializer=glorot_init)(y)
    y = SNDense(units=1, use_bias=True, kernel_initializer=glorot_init)(y)

    return Model(inputs=[x], outputs=[y, cls_out])



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
                self.current_discriminator_labels]

    def next_generator_sample_test(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size),
                (np.arange(self._batch_size) % 10).reshape((self._batch_size,1))]

    def _load_discriminator_data(self, index):
        self.current_discriminator_labels =  np.expand_dims(self._Y[index], axis=-1)
        return [self._X[index], self.current_discriminator_labels]

    def _shuffle_data(self):
        self._X, self._Y = shuffle(self._X, self._Y)

    def display(self, output_batch, input_batch=None):
        batch = output_batch[0]
        image = super(MNISTDataset, self).display(batch)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image

class ProjectiveGAN(GAN):
    def compile_intermediate_variables(self):
        self.generator_output = [self.generator(self.generator_input), self.generator_input[1]]
        self.discriminator_fake_output = self.discriminator(self.generator_output)
        self.discriminator_real_output = self.discriminator(self.discriminator_input)


def main():
    generator = make_generator_resnet_ci()
    discriminator = make_spectral_discriminator()

    generator.summary()
    discriminator.summary()

    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size)

    gan = AC_GAN(generator=generator, discriminator=discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()

if __name__ == "__main__":
    main()
