from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from gan import GAN
from dataset import ArrayDataset
from cmd import parser_with_default_args
from train import Trainer

import numpy as np
from layer_utils import resblock
from keras_contrib.layers import InstanceNormalization

from inception_score import get_inception_score

from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import os
from tqdm import tqdm

from spectral_normalized_layers import SNConv2D, SNDense

def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True, cache_dir='.')

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def make_generator():
    """Creates a generator model that takes a 128-dimensional noise vector as a "seed", and outputs images
    of size 128x64x3."""
    x = Input((128, ))
    y = Dense(128 * 4 * 4)(x)
    y = Reshape((4, 4, 128))(y)

    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'UP', 128)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias=False,
                      padding='same', activation='tanh')(y)
    return Model(inputs=x, outputs=y)


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((32, 32, 3))

    y = resblock(x, (3, 3), 'DOWN', 128, InstanceNormalization, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'DOWN', 128, InstanceNormalization, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'SAME', 128, InstanceNormalization, conv_layer=SNConv2D)
    y = resblock(y, (3, 3), 'SAME', 128, InstanceNormalization, conv_layer=SNConv2D)

    y = GlobalAveragePooling2D()(y)
    y = SNDense(units=1, use_bias=False)(y)

    return Model(inputs=x, outputs=y)


class CifarDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(128, )):
        (X_train, y_train), (X_test, y_test) = load_data()
        X = X_train# np.concatenate((X_train, X_test), axis=0)
        print (X.shape)
        X = (X.astype(np.float32) - 127.5) / 127.5
        super(CifarDataset, self).__init__(X, batch_size, noise_size)

    def display(self, output_batch, input_batch=None):
        batch = output_batch[0]
        image = super(CifarDataset, self).display(batch)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image


def main():
    generator = make_generator()
    discriminator = make_discriminator()

    print (generator.summary())
    print (discriminator.summary())

    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")
    args = parser.parse_args()

    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)

    args.generator_optimizer = Adam(args.lr, beta_1=0, beta_2=0.9)
    args.discriminator_optimizer = Adam(args.lr, beta_1=0, beta_2=0.9)

    def compute_inception_score():
        images = np.empty((50000, 32, 32, 3))
        dataset._batch_size = 100
        for i in tqdm(range(0, 50000, 100)):
            g_s = dataset.next_generator_sample_test()
            images[i:(i+100)] = generator.predict(g_s)
        images *= 127.5
        images += 127.5
        print(get_inception_score(images))
        dataset._batch_size = args.batch_size

    if args.phase == 'train':
        dataset = CifarDataset(args.batch_size)
        gan = GAN(generator=generator, discriminator=discriminator, **vars(args))
        trainer = Trainer(dataset, gan, at_store_checkpoint_hook = compute_inception_score,  lr_decay_shedule='linear', **vars(args))
        trainer.train()


if __name__ == "__main__":
    main()
