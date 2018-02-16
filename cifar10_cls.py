from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, UpSampling2D, Add, AveragePooling2D

from dataset import ArrayDataset
from cmd import parser_with_default_args
from train import Trainer

import numpy as np
from layer_utils import resblock
from keras_contrib.layers import InstanceNormalization

from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import os
from sklearn.utils import shuffle
from conditional_layers import ConditionalInstanceNormalization, ConditionalConv2D
from ac_gan import AC_GAN
from tqdm import tqdm
from inception_score import get_inception_score

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


def make_generator_ci():
    x = Input((128, ))
    cls = Input((1, ), dtype='int32')

    y = Dense(128 * 4 * 4)(x)
    y = Reshape((4, 4, 128))(y)

    conditional_instance_norm = lambda axis: (lambda inp: ConditionalInstanceNormalization(number_of_classes=10, axis=axis)([inp, cls]))

    y = resblock(y, (3, 3), 'UP', 128, conditional_instance_norm)
    y = resblock(y, (3, 3), 'UP', 128, conditional_instance_norm)
    y = resblock(y, (3, 3), 'UP', 128, conditional_instance_norm)

    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias=True,
                      padding='same', activation='tanh')(y)
    return Model(inputs=[x, cls], outputs=y)


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((32, 32, 3))

    y = resblock(x, (3, 3), 'DOWN', 128, norm=None, is_first=True)
    y = resblock(y, (3, 3), 'DOWN', 128, norm=None)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)
    y = resblock(y, (3, 3), 'SAME', 128, norm=None, conv_shortcut=False)

    y = Activation('relu')(y)

    y = GlobalAveragePooling2D()(y)
    cls_out = Dense(10, use_bias=True)(y)
    y = Dense(1, use_bias=True)(y)

    return Model(inputs=x, outputs=[y, cls_out])


class CifarDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(128, )):
        (X_train, y_train), (X_test, y_test) = load_data()
        X = X_train
        X = (X.astype(np.float32) - 127.5) / 127.5
        X += np.random.uniform(0, 1/128.0, size=X.shape)
        super(CifarDataset, self).__init__(X, batch_size, noise_size)
        self._Y = y_train
        self._cls_prob = np.bincount(np.squeeze(self._Y, axis=1)) / float(self._Y.shape[0])

    def number_of_batches_per_epoch(self):
        return 1000

    def number_of_batches_per_validation(self):
        return 10

    def next_generator_sample(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size),
                np.random.choice(10, size=(self._batch_size, 1), p=self._cls_prob)]

    def next_generator_sample_test(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size),
                (np.arange(self._batch_size) % 10).reshape((self._batch_size,1))]

    def _load_discriminator_data(self, index):
        return [self._X[index], self._Y[index]]

    def _shuffle_data(self):
        self._X, self._Y = shuffle(self._X, self._Y)

    def display(self, output_batch, input_batch=None):
        batch = output_batch[0]
        image = super(CifarDataset, self).display(batch)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image


def main():
    generator = make_generator_ci()
    discriminator = make_discriminator()

    print (generator.summary())
    print (discriminator.summary())

    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    args = parser.parse_args()

    def compute_inception_score():
        images = np.empty((50000, 32, 32, 3))
        dataset._batch_size = 100
        for i in tqdm(range(0, 50000, 100)):
            g_s = dataset.next_generator_sample()
            images[i:(i+100)] = generator.predict(g_s)
        images *= 127.5
        images += 127.5
        print(get_inception_score(images))
        dataset._batch_size = args.batch_size

    if args.phase == 'train':
        dataset = CifarDataset(args.batch_size)
        gan = AC_GAN(generator=generator, discriminator=discriminator, **vars(args))
        trainer = Trainer(dataset, gan, at_store_checkpoint_hook = compute_inception_score, **vars(args))
        trainer.train()



if __name__ == "__main__":
    main()
