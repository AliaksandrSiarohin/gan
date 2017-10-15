from keras.layers import Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.backend import tf as ktf
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


class LayerNorm(Layer):
    def __init__(self, epsilon=1e-7, beta_init='zero', gamma_init='one', **kwargs):
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        
        self.epsilon = epsilon
        self.uses_learning_phase = True
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1],)
        
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        
        self.trainable_weights = [self.gamma, self.beta]

    def call(self, x, mask=None):
        m, std = ktf.nn.moments(x, axes = [1, 2, 3], keep_dims=True)
        x_normed = (x - m) / (K.sqrt(std + self.epsilon))
        out = self.gamma * x_normed + self.beta
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {"epsilon": self.epsilon}
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    

def resblock(x, kernel_size, resample, nfilters, norm = BatchNormalization):
    assert resample in ["UP", "SAME", "DOWN"]

    if resample == "UP":
        shortcut = UpSampling2D(size=(2, 2)) (x)        
        shortcut = Conv2D(nfilters, kernel_size, padding = 'same',
                          kernel_initializer='he_uniform', use_bias = True) (shortcut)
                
        convpath = norm() (x)
        convpath = Activation('relu') (convpath)
        convpath = UpSampling2D(size=(2, 2))(convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform', 
                                      use_bias = False, padding='same')(convpath)
        convpath = norm() (convpath)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                     use_bias = True, padding='same') (convpath)
        
        y = Add() ([shortcut, convpath])
    elif resample == "SAME":      
        shortcut = Conv2D(nfilters, kernel_size, padding = 'same',
                          kernel_initializer='he_uniform', use_bias = True) (x)
                
        convpath = norm() (x)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform', 
                                 use_bias = False, padding='same')(convpath)        
        convpath = norm() (convpath)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = True, padding='same') (convpath)
        
        y = Add() ([shortcut, convpath])
        
    else:
        shortcut = AveragePooling2D(pool_size = (2, 2)) (x)
        shortcut = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                          padding = 'same', use_bias = True) (shortcut)        
        
        convpath = x
        convpath = norm() (x)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = False, padding='same')(convpath)
        convpath = AveragePooling2D(pool_size = (2, 2)) (convpath)
        convpath = norm() (convpath)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = True, padding='same') (convpath)        
        y = Add() ([shortcut, convpath])
        
    return y
