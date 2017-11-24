from keras.models import Model, Input
from keras.layers import Activation, GlobalAveragePooling2D, Add
from keras.layers import Dense, Conv2D
import keras.backend as K
from keras.applications import inception_v3

from layer_utils import resblock
import numpy as np

from cmd import parser_with_default_args
from dataset import FolderDataset
from wgan_gp import WGAN_GP
from train import Trainer


def make_generator(image_size):
    inp = Input(list(image_size) + [3])
    scores = Input([1])

    out = Conv2D(32, (3, 3), use_bias=True, padding='same')(inp)
    out = resblock(out, (3, 3), 'DOWN', 64)
    out = resblock(out, (3, 3), 'DOWN', 128)
    out = resblock(out, (3, 3), 'UP', 128)
    out = resblock(out, (3, 3), 'UP', 64)

    out = Conv2D(3, (3, 3), use_bias=True, padding='same')(out)
    out = Activation('tanh')(out)
    return Model(inputs=[inp, scores], outputs=[out, scores])


def make_discriminator(image_size):
    inp = Input(list(image_size) + [3])
    scores = Input([1])

    model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_tensor=inp)

    out = model(inp)

    out = GlobalAveragePooling2D()(out)
    out = Dense(1)(out)
    out = Add()([out, scores])
    return Model(inputs=[inp, scores], outputs=out)

class MEM_GAN(WGAN_GP):
    def __init__(self, generator, discriminator, l1_penalty_weight=100, **kwargs):
        super(MEM_GAN, self).__init__(generator, discriminator, **kwargs)
        self.generator_metric_names = ['l1', 'gan_loss', 'mem_inc']
        self.discriminator_metric_names = ['true', 'fake']
        self.l1_penalty_weight = l1_penalty_weight

    def _compile_generator_loss(self):
        fake = self._discriminator_fake_input

        l1 = self.l1_penalty_weight * K.mean(K.abs(self._generator_input[0] - fake[0]))

        def l1_loss(y_true, y_pred):
            return l1

        def mem_inc(y_true, y_pred):
            return K.mean((y_pred))

        def mem_loss(y_true, y_pred):
            return K.mean((y_pred - fake[1] - 1) ** 2)

        def generator_least_square_loss(y_true, y_pred):
            return mem_loss(y_pred, y_pred) + l1_loss(y_true, y_pred)
        return generator_least_square_loss, [l1_loss, mem_loss, mem_inc]

    def _compile_discriminator_loss(self):
        _, metrics = super(MEM_GAN, self)._compile_discriminator_loss()
        gp_fn_list = metrics[0:1]

        fake = self._discriminator_fake_input

        def true_loss(y_true, y_pred):
            y_true = y_pred[:K.shape(y_true)[0]]
            return K.mean(y_true ** 2)

        def fake_loss(y_true, y_pred):
            y_fake = y_pred[K.shape(y_true)[0]:] - fake[1]
            return K.mean(y_fake ** 2)


        def gp_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), gp_fn_list), K.zeros((1, )))

        def loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred)# + gp_loss(y_true, y_pred)

        return loss, [true_loss, fake_loss]

import os
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.color import gray2rgb

class LamemDataset(FolderDataset):
    def __init__(self, input_dir, batch_size, image_size, train_file):
        super(FolderDataset, self).__init__(batch_size, None)
        with open(train_file) as f:
            image_score_pair = [pair.split(' ') for pair in f.read().split('\n')]
            image_score_pair = filter(lambda x: len(x) == 2, image_score_pair)
            image_score_pair = map(lambda x: (str(x[0]), float(x[1])), image_score_pair)
            self.image_score_pairs = np.array(image_score_pair)

        self._input_dir = input_dir
        self._image_size = image_size
        self._batches_before_shuffle = int(self.image_score_pairs.shape[0] // self._batch_size)

    def number_of_batches_per_epoch(self):
        return 1000

    def _preprocess_image(self, img):
        if len(img.shape) == 2:
            img = img_as_ubyte(gray2rgb(img))
        return (img/255.0) * 2 - 1

    def _deprocess_image(self, img):
        return img_as_ubyte((img + 1) / 2)

    def _load_data_batch(self, index):
        images = np.array([self._preprocess_image(imread(os.path.join(self._input_dir, img_name)))
                          for img_name in self.image_score_pairs[index, 0]])
        scores = -np.array(self.image_score_pairs[index,1], dtype='float32')
        data = [images, scores]
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
        np.random.shuffle(self.image_score_pairs)

    def display(self, output_batch, input_batch = None):
        image_in = super(FolderDataset, self).display(input_batch[0])
        image_out = super(FolderDataset, self).display(output_batch[0])
        return self._deprocess_image(np.concatenate([image_in, image_out], axis=1))



def main():
    parser = parser_with_default_args()
    parser.add_argument("--input_dir", default='lamem/image_r',
                        help='Foldet with input images')
    parser.add_argument("--train_file", default='lamem/splits/train_1.txt', help="File with name and scores")

    args = parser.parse_args()
    args.batch_size = 4
    args.training_ratio = 1
    image_size = (256, 256)

    generator = make_generator(image_size)

    discriminator = make_discriminator(image_size)

    dataset = LamemDataset(args.input_dir, args.batch_size, image_size, args.train_file)
    gan_type = MEM_GAN
    gan = gan_type(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()

if __name__ == "__main__":
    main()
