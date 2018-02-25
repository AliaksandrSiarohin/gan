from keras.layers import Conv2D, Dense
import keras.initializers
from keras import backend as K
from keras.backend import tf as ktf
from conditional_layers import ConditionalConv11, ConditionalDense
import numpy as np
from keras.initializers import RandomNormal


def max_singular_val(w, u, fully_differentiable=False, ip=1, transpose=lambda x: K.transpose(x)):
    if not fully_differentiable:
        w_ = K.stop_gradient(w)
    else:
        w_ = w
    u = K.expand_dims(u, axis=-1)

    u_bar = u
    for _ in range(ip):
        v_bar = ktf.matmul(transpose(w_), u_bar)
        v_bar = K.l2_normalize(v_bar, axis=(-1, -2))

        u_bar_raw = ktf.matmul(w_, v_bar)
        u_bar = K.l2_normalize(u_bar_raw, axis=(-1, -2))
    sigma = ktf.matmul(transpose(u_bar), ktf.matmul(w, v_bar))

    sigma = K.squeeze(sigma, axis=-1)
    sigma = K.squeeze(sigma, axis=-1)

    u_bar = K.squeeze(u_bar, axis=-1)
    return sigma, u_bar


def max_singular_val_for_convolution(w, u, fully_differentiable=False, ip=1, padding='same',
                                     strides=(1, 1), data_format='channels_last'):
    assert ip >= 1
    if not fully_differentiable:
        w_ = K.stop_gradient(w)
    else:
        w_ = w

    u_bar = u
    for _ in range(ip):
        v_bar = K.conv2d(u_bar, w_, strides=strides, data_format=data_format, padding=padding)
        v_bar = K.l2_normalize(v_bar)

        u_bar_raw = K.conv2d_transpose(v_bar, w_, output_shape=K.int_shape(u),
                                       strides=strides, data_format=data_format, padding=padding)
        u_bar = K.l2_normalize(u_bar_raw)

    u_bar_raw_diff = K.conv2d_transpose(v_bar, w, output_shape=K.int_shape(u),
                                        strides=strides, data_format=data_format, padding=padding)
    sigma = K.sum(u_bar * u_bar_raw_diff)
    return sigma, u_bar


class SNConv2D(Conv2D):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), conv_singular=False,
                 fully_diff_spectral=True, spectral_iterations=1, **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.conv_singular = conv_singular
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = True
        super(SNConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SNConv2D, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.conv_singular:
            self.u = self.add_weight(
                shape=(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(1, input_shape[1], input_shape[2], input_shape[3]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        if self.conv_singular:
            sigma, u_bar = max_singular_val_for_convolution(self.kernel, self.u,
                                                            fully_differentiable=self.fully_diff_spectral,
                                                            ip=self.spectral_iterations,
                                                            padding=self.padding,
                                                            strides=self.strides, data_format=self.data_format)
            kernel_sn = self.kernel / sigma
            self.add_update(K.update(self.u, u_bar))
        else:
            kernel_shape = K.int_shape(self.kernel)
            w = K.reshape(self.kernel, (kernel_shape[0] * kernel_shape[1] * kernel_shape[2], kernel_shape[3]))

            sigma, u_bar = max_singular_val(w, self.u, fully_differentiable=self.fully_diff_spectral,
                                            ip=self.spectral_iterations,)

            w_sn = w / sigma

            kernel_sn = K.reshape(w_sn, kernel_shape)

            self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNConv2D, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNDense(Dense):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = True
        super(SNDense, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SNDense, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.u = self.add_weight(
            shape=(kernel_shape[0], ),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)

    def call(self, inputs):
        w = self.kernel
        sigma, u_bar = max_singular_val(w, self.u, fully_differentiable=self.fully_diff_spectral,
                                            ip=self.spectral_iterations)
        w_sn = w / sigma
        kernel_sn = w_sn
        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNDense, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNConditionalConv11(ConditionalConv11):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = True
        super(SNConditionalConv11, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SNConditionalConv11, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.u = self.add_weight(
            shape=(self.number_of_classes, kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)

    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)
        w = K.reshape(self.kernel, (kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3], kernel_shape[-1]))
        sigma, u_bar = max_singular_val(w, self.u, transpose=lambda x: ktf.transpose(x, [0, 2, 1]),
                                        fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
        sigma = K.reshape(sigma, (self.number_of_classes, 1, 1, 1, 1))
        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNConditionalConv11, self).call(inputs)
        self.kernel = kernel

        return outputs

class SNCondtionalDense(ConditionalDense):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = True
        super(SNCondtionalDense, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SNCondtionalDense, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.u = self.add_weight(
            shape=(self.number_of_classes, kernel_shape[1]),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)


    def call(self, inputs):
        w = self.kernel
        sigma, u_bar = max_singular_val(w, self.u, transpose=lambda x: ktf.transpose(x, [0, 2, 1]),
                                        fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
        sigma = K.reshape(sigma, (self.number_of_classes, 1, 1))
        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNCondtionalDense, self).call(inputs)
        self.kernel = kernel

        return outputs


def test_conv_with_conv_spectal():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((3, 3, 1))
    out = SNConv2D(kernel_size=(2, 2), padding='valid', filters=1, kernel_initializer=kernel_init, conv_singular=True)(inp)
    m = Model([inp], [out])
    x = np.arange(3 * 3).reshape((1, 3, 3, 1))
    for i in range(100):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    matrix = np.zeros((9, 4))
    matrix[[0, 1, 3, 4], 0] = kernel.reshape((-1, ))
    matrix[[1, 2, 4, 5], 1] = kernel.reshape((-1, ))
    matrix[[3, 4, 6, 7], 2] = kernel.reshape((-1, ))
    matrix[[4, 5, 7, 8], 3] = kernel.reshape((-1, ))

    _, s, _ = svd(matrix)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val_for_convolution(w, u, ip=1, padding='valid')[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_singular_val_for_convolution():
    from numpy.linalg import svd
    w = K.placeholder([2, 2, 1, 1])
    u = K.placeholder([1, 3, 3, 1])
    v = K.placeholder([1, 2, 2, 1])

    f = K.function([w, u], max_singular_val_for_convolution(w, u, ip=100, padding='valid'))

    #conv = K.function([w, u], [K.conv2d(u, w, strides=(1,1), data_format='channels_last', padding='valid')])
    #conv_tr = K.function([w, v], [K.conv2d_transpose(v, w, strides=(1,1), output_shape=K.int_shape(u),
    #                                                 data_format='channels_last', padding='valid')])

    #np.random.seed(0)
    w_values = np.random.normal(size=[2, 2, 1, 1])
    u_values = np.random.normal(size=[1, 3, 3, 1])

    matrix = np.zeros((9, 4))

    matrix[[0, 1, 3, 4], 0] = w_values.reshape((-1, ))
    matrix[[1, 2, 4, 5], 1] = w_values.reshape((-1, ))
    matrix[[3, 4, 6, 7], 2] = w_values.reshape((-1, ))
    matrix[[4, 5, 7, 8], 3] = w_values.reshape((-1, ))

    _, s, _ = svd(matrix)
    sigma, u = f([w_values, u_values])

    assert np.abs(sigma - s[0]) < 1e-5


def test_dense():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((5, ))
    out = SNDense(units=10, kernel_initializer=kernel_init)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 10).reshape((10, 5))
    for i in range(50):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_iterations():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((5, ))
    out = SNDense(units=10, kernel_initializer=kernel_init, spectral_iterations=50)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 10).reshape((10, 5))
    for i in range(1):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_conv2D():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    out = SNConv2D(kernel_size=(3, 3), padding='same', filters=10, kernel_initializer=kernel_init)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    for i in range(100):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    kernel = kernel.reshape((-1, kernel.shape[3]))

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_conditional_conv():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    cls = Input((1, ), dtype='int32')
    out = SNConditionalConv11(number_of_classes=3, filters=10, kernel_initializer=kernel_init)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel_all = K.get_value(m.layers[2].kernel)
    u_val_all = K.get_value(m.layers[2].u)

    for i in range(3):
        kernel = kernel_all[i]
        kernel = kernel.reshape((-1, kernel.shape[3]))
        u_val = u_val_all[i]

        _, s, _ = svd(kernel)

        w = K.placeholder(kernel.shape)
        u = K.placeholder(u_val.shape)
        max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

        assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_conditional_dense():
    from keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input((4, ))
    cls = Input((1, ), dtype='int32')
    out = SNCondtionalDense(number_of_classes=3, units=10, kernel_initializer=kernel_init)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 4).reshape((5, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel_all = K.get_value(m.layers[2].kernel)
    u_val_all = K.get_value(m.layers[2].u)

    for i in range(3):
        kernel = kernel_all[i]
        kernel = kernel.reshape((-1, kernel.shape[-1]))
        u_val = u_val_all[i]

        _, s, _ = svd(kernel)

        w = K.placeholder(kernel.shape)
        u = K.placeholder(u_val.shape)
        max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])
        assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


if __name__ == "__main__":
    test_conv_with_conv_spectal()
    test_conditional_dense()
    test_conditional_conv()
    test_conv2D()
    test_dense()
    test_singular_val_for_convolution()
    test_conv_with_conv_spectal()
    test_iterations()
