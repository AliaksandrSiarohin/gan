from keras.models import Model, Input
from keras.layers import Activation, GlobalAveragePooling2D, Flatten
from keras.applications import InceptionV3
from keras.layers import Dense, Conv2D

from layer_utils import resblock
import numpy as np

from cmd import parser_with_default_args
from dataset import FolderDataset
from wgan_gp import WGAN_GP
from train import Trainer


def make_generator(image_size):
    inp = Input(list(image_size) + [3])

    out = Conv2D(32, (3, 3), use_bias=True, padding='same')(inp)
    out = resblock(out, (3, 3), 'DOWN', 64)
    out = resblock(out, (3, 3), 'DOWN', 128)
    out = resblock(out, (3, 3), 'UP', 128)
    out = resblock(out, (3, 3), 'UP', 64)

    out = Conv2D(3, (3, 3), use_bias=True, padding='same')(out)
    out = Activation('tanh') (out)
    return Model(inputs=inp, outputs=out)


def make_discriminator(image_size):
    inp = Input(list(image_size) + [3])

    model = InceptionV3(weights='imagenet', include_top=False, input_tensor=inp)

    out = model(inp)

    out = GlobalAveragePooling2D()(out)
    out = Dense(1)(out)
    return Model(inp, out)


import os
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.io import imread

class LamemDataset(FolderDataset):
    def __init__(self, input_dir, batch_size, image_size, train_file):
        super(FolderDataset, self).__init__(batch_size, None)
        with open(train_file) as f:
            image_score_pair = [pair.split(' ') for pair in f.read().split('\n')]
            image_score_pair = filter(lambda x: len(x) == 2, image_score_pair)
            self._image_names = np.array(map(lambda x: x[0], image_score_pair))

        self._input_dir = input_dir
        self._image_size = image_size
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)

    def number_of_batches_per_epoch(self):
        return 1000

    def _preprocess_image(self, img):
        return resize(img, self._image_size) * 2 - 1

    def _deprocess_image(self, img):
        return img_as_ubyte((img + 1) / 2)

    def _load_data_batch(self, index):
        data = [np.array([self._preprocess_image(imread(os.path.join(self._input_dir, img_name)))
                          for img_name in self._image_names[index]])]
        return data

    def next_generator_sample(self):
        index = self._next_data_index()
        image_batch = self._load_data_batch(index)
        return image_batch

    def next_discriminator_sample(self):
        index = self._next_data_index()
        image_batch = self._load_data_batch(index)
        return image_batch

    def _shuffle_data(self):
        np.random.shuffle(self._image_names)

    def display(self, output_batch, input_batch = None):
        image_in = super(FolderDataset, self).display(input_batch[0])
        image_out = super(FolderDataset, self).display(output_batch)
        return self._deprocess_image(np.concatenate([image_in, image_out], axis=1))

def main():
    parser = parser_with_default_args()
    parser.add_argument("--input_dir", default='lamem/images',
                        help='Foldet with input images')
    parser.add_argument("--train_file", default='lamem/splits/train_1.txt', help="File with name and scores")

    args = parser.parse_args()
    args.batch_size = 1
    args.training_ratio = 5
    image_size = (256, 256)

    generator = make_generator(image_size)

    discriminator = make_discriminator(image_size)

    dataset = LamemDataset(args.input_dir, args.batch_size, image_size, args.train_file)
    gan_type = WGAN_GP
    gan = gan_type(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()

if __name__ == "__main__":
    main()
