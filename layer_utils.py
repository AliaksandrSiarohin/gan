from keras.layers import Activation, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D
from keras.backend import tf as ktf
from keras.engine.topology import Layer
from keras.models import Input, Model
from keras.layers.pooling import _GlobalPooling2D
from keras import backend as K
from functools import partial
from keras.layers import LeakyReLU

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


class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])

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


def resblock(x, kernel_size, resample, nfilters, name, norm=BatchNormalization, is_first=False, conv_layer=Conv2D):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1

    identity = lambda x: x

    if norm is None:
        norm = lambda axis, name: identity

    if resample == "UP":
        resample_op = UpSampling2D(size=(2, 2), name=name + '_up')
    elif resample == "DOWN":
        resample_op = AveragePooling2D(pool_size=(2, 2), name=name + '_pool')
    else:
        resample_op = identity

    in_filters = K.int_shape(x)[feature_axis]

    if resample == "SAME" and in_filters == nfilters:
        shortcut_layer = identity
    else:
        shortcut_layer = conv_layer(kernel_size=(1, 1), filters=nfilters, kernel_initializer=he_init, name=name + 'shortcut')

    ### SHORTCUT PAHT
    if is_first:
        shortcut = resample_op(x)
        shortcut = shortcut_layer(shortcut)
    else:
        shortcut = shortcut_layer(x)
        shortcut = resample_op(shortcut)

    ### CONV PATH
    convpath = x
    if not is_first:
        convpath = norm(axis=feature_axis, name=name + '_bn1')(convpath)
        convpath = Activation('relu')(convpath)
    if resample == "UP":
        convpath = resample_op(convpath)

    convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                                      use_bias=True, padding='same', name=name + '_conv1')(convpath)

    convpath = norm(axis=feature_axis, name=name + '_bn2')(convpath)
    convpath = Activation('relu')(convpath)

    convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                          use_bias=True, padding='same', name=name + '_conv2')(convpath)

    if resample == "DOWN":
        convpath = resample_op(convpath)

    y = Add()([shortcut, convpath])

    return y


def dcblock(x, kernel_size, resample, nfilters, name, norm=BatchNormalization, is_first=False, conv_layer=Conv2D):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1

    convpath = x
    if resample == "UP":
        convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
        convpath = Activation('relu', name=name + 'relu')(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, strides=(2, 2),
                              name=name + '.conv', padding='same')(convpath)
    elif resample == "SAME":
       if not is_first:
           convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
           convpath = LeakyReLU(name=name + 'relu')(convpath)

       convpath = conv_layer(filters=nfilters, kernel_size=kernel_size,
                              name=name + '.conv', padding='same')(convpath)
    elif resample == "DOWN":
        if not is_first:
            convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
            convpath = LeakyReLU(name=name + 'relu')(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, strides=(2, 2),
                              name=name + '.conv', padding='same')(convpath)
    return convpath
