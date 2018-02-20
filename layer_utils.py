from keras.layers import Activation, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D
from keras.backend import tf as ktf
from keras.engine.topology import Layer
from keras.models import Input, Model
from keras import backend as K
from functools import partial

import numpy as np

def jacobian(y_flat, x):
    n = y_flat.shape[0]

    loop_vars = [
        ktf.constant(0, ktf.int32),
        ktf.TensorArray(ktf.float32, size=n),
    ]

    _, jacobian = ktf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, ktf.gradients(y_flat[j], x))),
        loop_vars)

    return jacobian.stack()


def content_features_model(image_size, layer_name='block4_conv1'):
    from keras.applications import vgg19
    x = Input(list(image_size) + [3])
    def preprocess_for_vgg(x):
        x = 255 * (x + 1) / 2
        mean = np.array([103.939, 116.779, 123.68])
        mean = mean.reshape((1, 1, 1, 3))
        x = x - mean
        x = x[..., ::-1]
        return x

    x = Input((128, 64, 3))
    y = Lambda(preprocess_for_vgg)(x)
    vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=y)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    if type(layer_name) == list:
        y = [outputs_dict[ln] for ln in layer_name]
    else:
        y = outputs_dict[layer_name]
    return Model(inputs=x, outputs=y)



class GaussianFromPointsLayer(Layer):
    def __init__(self, sigma=6, image_size=(128, 64), **kwargs):
        self.sigma = sigma
        self.image_size = image_size
        super(GaussianFromPointsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.xx, self.yy = ktf.meshgrid(ktf.range(self.image_size[1]),
                                        ktf.range(self.image_size[0]))
        self.xx = ktf.expand_dims(ktf.cast(self.xx, 'float32'), 2)
        self.yy = ktf.expand_dims(ktf.cast(self.yy, 'float32'), 2)

    def call(self, x, mask=None):
        def batch_map(cords):
            y = ((cords[..., 0] + 1.0) / 2.0) * self.image_size[0]
            x = ((cords[..., 1] + 1.0) / 2.0) * self.image_size[1]
            y = ktf.reshape(y, (1, 1, -1))
            x = ktf.reshape(x, (1, 1, -1))
            return ktf.exp(-((self.yy - y) ** 2 + (self.xx - x) ** 2) / (2 * self.sigma ** 2))

        x = ktf.map_fn(batch_map, x, dtype='float32')
        print (x.shape)
        return x

    def compute_output_shape(self, input_shape):
        print (input_shape)
        return tuple([input_shape[0], self.image_size[0], self.image_size[1], input_shape[1]])

    def get_config(self):
        config = {"sigma": self.sigma, "image_size": self.image_size}
        base_config = super(GaussianFromPointsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def uniform_init(shape, constant = 4.0):
    if len(shape) >= 4:
        stdev = np.sqrt(constant / ((shape[1] ** 2) * (shape[-1] + shape[-2])))
    else:
        stdev = np.sqrt(constant / (shape[0] + shape[1]))
    return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=shape
            ).astype('float32')

he_init = partial(uniform_init, constant=4.0)
glorot_init = partial(uniform_init, constant=2.0)


def resblock(x, kernel_size, resample, nfilters, norm=BatchNormalization, is_first=False, conv_shortcut=True,
             conv_layer=Conv2D):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if norm is None:
        norm = lambda axis: lambda x: x ##Identity, no normalization

    if resample == "UP":
        shortcut = UpSampling2D(size=(2, 2)) (x)
        shortcut = conv_layer(filters=nfilters, kernel_size=(1, 1), padding = 'same',
                          kernel_initializer=glorot_init, use_bias = True) (shortcut)

        convpath = x
        if not is_first:
            convpath = norm(axis=feature_axis)(convpath)
            convpath = Activation('relu') (convpath)
        convpath = UpSampling2D(size=(2, 2))(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size,
                              kernel_initializer=he_init, use_bias=True, padding='same')(convpath)
        convpath = norm(axis=feature_axis)(convpath)
        convpath = Activation('relu')(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                              use_bias=True, padding='same') (convpath)

        y = Add() ([shortcut, convpath])
    elif resample == "SAME":
        if conv_shortcut:
            shortcut = conv_layer(filters=nfilters, kernel_size=(1, 1), padding = 'same',
                              kernel_initializer=glorot_init, use_bias = True) (x)
        else:
            shortcut = x

        convpath = x
        if not is_first:
            convpath = norm(axis=feature_axis)(convpath)
            convpath = Activation('relu') (convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                               use_bias=True, padding='same')(convpath)
        convpath = norm(axis=feature_axis)(convpath)
        convpath = Activation('relu') (convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                                 use_bias=True, padding='same') (convpath)

        y = Add() ([shortcut, convpath])

    else:
        if not is_first:
            shortcut = conv_layer(filters=nfilters, kernel_size=(1, 1), kernel_initializer=glorot_init,
                              padding = 'same', use_bias=True) (x)
            shortcut = AveragePooling2D(pool_size=(2, 2))(shortcut)
        else:
            shortcut = AveragePooling2D(pool_size=(2, 2))(x)
            shortcut = conv_layer(filters=nfilters, kernel_size=(1, 1), kernel_initializer=he_init,
                                  padding = 'same', use_bias=True) (shortcut)


        convpath = x
        if not is_first:
            convpath = norm(axis=feature_axis)(convpath)
            convpath = Activation('relu') (convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                                 use_bias=True, padding='same')(convpath)
        if not is_first:
            convpath = norm(axis=feature_axis)(convpath)
        convpath = Activation('relu')(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                                 use_bias=True, padding='same') (convpath)
        convpath = AveragePooling2D(pool_size = (2, 2))(convpath)
        y = Add() ([shortcut, convpath])
        
    return y


