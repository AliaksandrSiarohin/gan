from keras.layers import Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D
from keras.backend import tf as ktf


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


def resblock(x, kernel_size, resample, nfilters, norm=BatchNormalization):
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
