from keras.layers import Conv2D, Dense
import keras.initializers
from keras import backend as K
from keras.backend import tf as ktf
from conditional_layers import ConditionalConv11, ConditionalDense
import numpy as np


def max_singular_val(w, u, transpose=lambda x: K.transpose(x)):
    u = K.expand_dims(u, axis=-1)
    v_bar = ktf.matmul(transpose(w), u)
    v_bar = K.l2_normalize(v_bar, axis=(-1, -2))

    u_bar_raw = ktf.matmul(w, v_bar)
    u_bar = K.l2_normalize(u_bar_raw, axis=(-1, -2))
    sigma = ktf.matmul(transpose(u_bar), u_bar_raw)

    sigma = K.squeeze(sigma, axis=-1)
    sigma = K.squeeze(sigma, axis=-1)

    #sigma = K.stop_gradient(sigma)

    u_bar = K.squeeze(u_bar, axis=-1)
    return sigma, u_bar


class SNConv2D(Conv2D):
    def __init__(self, sigma_initializer='glorot_uniform', **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        super(SNConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SNConv2D, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.u = self.add_weight(
            shape=(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], ),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)


    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)
        w = K.reshape(self.kernel, (kernel_shape[0] * kernel_shape[1] * kernel_shape[2], kernel_shape[3]))

        sigma, u_bar = max_singular_val(w, self.u)

        w_sn = w / sigma

        kernel_sn = K.reshape(w_sn, kernel_shape)

        self.updates.append(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNConv2D, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNDense(Dense):
    def __init__(self, sigma_initializer='glorot_uniform', **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
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
        sigma, u_bar = max_singular_val(w, self.u)
        w_sn = w / sigma
        kernel_sn = w_sn
        self.updates.append(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNDense, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNConditionalConv11(ConditionalConv11):
    def __init__(self, sigma_initializer='glorot_uniform', **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
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
        sigma, u_bar = max_singular_val(w, self.u, transpose=lambda x: ktf.transpose(x, [0, 2, 1]))
        sigma = K.reshape(sigma, (self.number_of_classes, 1, 1, 1, 1))
        self.updates.append(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNConditionalConv11, self).call(inputs)
        self.kernel = kernel

        return outputs

class SNCondtionalDense(ConditionalDense):
    def __init__(self, sigma_initializer='glorot_uniform', **kwargs):
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
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
        sigma, u_bar = max_singular_val(w, self.u, transpose=lambda x: ktf.transpose(x, [0, 2, 1]))
        sigma = K.reshape(sigma, (self.number_of_classes, 1, 1))
        self.updates.append(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNCondtionalDense, self).call(inputs)
        self.kernel = kernel

        return outputs


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
        K.get_session().run(m.layers[1].updates, feed_dict={inp:x})

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
        K.get_session().run(m.layers[1].updates, feed_dict={inp:x})

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
        K.get_session().run(m.layers[2].updates, feed_dict={inp:x, cls:cls_val})

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
        return np.random.normal(size=shape)

    inp = Input((4, ))
    cls = Input((1, ), dtype='int32')
    out = SNCondtionalDense(number_of_classes=3, units=10, kernel_initializer=kernel_init)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 4).reshape((5, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        K.get_session().run(m.layers[2].updates, feed_dict={inp:x, cls:cls_val})

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
    test_conditional_conv()
