from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.backend import tf as ktf
from keras.utils import conv_utils
from keras import activations
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, Add, AveragePooling2D, Reshape

from layer_utils import he_init, glorot_init

class ConditionalInstanceNormalization(Layer):
    """Conditional Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    Each class has it own normalization parametes.
    # Arguments
        number_of_classes: Number of classes, 10 for cifar10.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)
    """
    def __init__(self,
                 number_of_classes,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)
        self.number_of_classes = number_of_classes
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape[0])
        cls = input_shape[1]
        if len(cls) != 2:
            raise ValueError("Classes should be one dimensional")

        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        if self.axis is None:
            shape = (self.number_of_classes, 1)
        else:
            shape = (self.number_of_classes, input_shape[0][self.axis])

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        super(ConditionalInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        class_labels = K.squeeze(inputs[1], axis=1)
        inputs = inputs[0]
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        #del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        #broadcast_shape[0] = K.shape(inputs)[0]
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_shape[0] = K.shape(inputs)[0]
            broadcast_gamma = K.reshape(K.gather(self.gamma, class_labels), broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_shape[0] = K.shape(inputs)[0]
            broadcast_beta = K.reshape(K.gather(self.beta, class_labels), broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalConv11(Layer):
    def __init__(self, filters,
             number_of_classes,
             strides=1,
             data_format=None,
             activation=None,
             use_bias=True,
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros',
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             kernel_constraint=None,
             bias_constraint=None,
             **kwargs):
        super(ConditionalConv11, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple((1, 1), 2, 'kernel_size')
        self.number_of_classes = number_of_classes
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding('same')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(1, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        self.input_dim = input_dim
        kernel_shape = (self.number_of_classes, ) + self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.number_of_classes, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(ConditionalConv11, self).build(input_shape)

    def call(self, inputs):
        cls = inputs[1]
        x = inputs[0]

        _, w, h, in_c = K.int_shape(x)
        ### Preprocess input
        #(bs, w, h, c)
        if self.data_format != 'channels_first':
            x = ktf.transpose(x,  [0, 3, 1, 2])
        #(bs, c, w, h)
        x = ktf.reshape(x, (-1, in_c, w * h))
        #(bs, c, w*h)
        x = ktf.transpose(x, [0, 2, 1])
        #(bs, w*h, c)

        ### Preprocess filter
        cls = ktf.squeeze(cls, axis=1)
        #(num_cls, 1, 1, in, out)
        kernel = ktf.gather(self.kernel, cls)
        #(bs, 1, 1, in, out)

        kernel = ktf.squeeze(kernel, axis=1)
        kernel = ktf.squeeze(kernel, axis=1)
        #print (K.int_shape(kernel))
        #(in, 1, bs, out)
        #print (K.int_shape(kernel))

        output = ktf.matmul(x, kernel)
        #(bs, w*h, out)

        ### Deprocess output
        output = ktf.transpose(output, [0, 2, 1])
        # (bs, out, w * h)
        output = ktf.reshape(output, (-1, self.filters, w, h))
        # (bs, out, w, h)
        if self.bias is not None:
            #(num_cls, out)
            bias = ktf.gather(self.bias, cls)
            #(bs, bias)
            bias = ktf.expand_dims(bias, axis=-1)
            bias = ktf.expand_dims(bias, axis=-1)
            #(bs, bias, 1, 1)
            output += bias

        if self.data_format != 'channels_first':
            #(bs, out, w, h)
            output = ktf.transpose(output, [0, 2, 3, 1])

        if self.activation is not None:
            return self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'rank': 2,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConditionalConv11, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalConv2D(Layer):
    def __init__(self, filters,
             kernel_size,
             number_of_classes,
             strides=1,
             padding='valid',
             data_format=None,
             dilation_rate=1,
             activation=None,
             use_bias=True,
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros',
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             kernel_constraint=None,
             bias_constraint=None,
             **kwargs):
        super(ConditionalConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.number_of_classes = number_of_classes
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = (self.number_of_classes, ) + self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.number_of_classes, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(ConditionalConv2D, self).build(input_shape)

    def call(self, inputs):
        def apply_separate_filter_for_each_batch(inputs):
            kernel = inputs[1]
            x = K.expand_dims(inputs[0], axis=0)
            outputs = K.conv2d(
                        x,
                        kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
            if self.bias is not None:
                bias = inputs[2]
                outputs = K.bias_add(outputs, bias, data_format=self.data_format)
            return K.squeeze(outputs, axis=0)
        x = inputs[0]
        classes = K.squeeze(inputs[1], axis=1)

        if self.bias is not None:
            outputs = K.map_fn(apply_separate_filter_for_each_batch,
                          [x, K.gather(self.kernel, classes), K.gather(self.bias, classes)], dtype='float32')
        else:
            outputs = K.map_fn(apply_separate_filter_for_each_batch,
                          [x, K.gather(self.kernel, classes)], dtype='float32')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'rank': 2,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConditionalConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalDense(Layer):
    def __init__(self, units,
                 number_of_classes,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ConditionalDense, self).__init__(**kwargs)
        self.units = units
        self.number_of_classes = number_of_classes
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = input_shape[0]
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.number_of_classes, input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.number_of_classes, self.units),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        classes = K.squeeze(inputs[1], axis=1)
        kernel = K.gather(self.kernel, classes)
        #(bs, in, out)

        x = K.expand_dims(inputs[0], axis=1)
        #(bs, 1, in)
        output = ktf.matmul(x, kernel)
        #(bs, 1, out)
        output = K.squeeze(output, axis=1)
        #(bs, out)

        if self.bias is not None:
            b = K.gather(self.bias, classes)
            output += b

        if self.activation is not None:
            return self.activation(output)
        return output


    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConditionalDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def cond_resblock(x, cls, kernel_size, resample, nfilters, number_of_classes,
                  norm=BatchNormalization, is_first=False, conv_shortcut=True, conv_layer=Conv2D,
                  cond_bottleneck_layer=ConditionalConv11, uncond_bottleneck_layer=Conv2D):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1

    identity = lambda x: x

    if norm is None:
        norm = lambda axis: identity

    if resample == "UP":
        resample_op = UpSampling2D(size=(2, 2))
    elif resample == "DOWN":
        resample_op = AveragePooling2D(pool_size=(2, 2))
    else:
        resample_op = identity


    ### SHORTCUT PAHT
    shortcut = x
    if conv_shortcut:
        shortcut = conv_layer(filters=nfilters, kernel_size=(1, 1), padding='same',
                          kernel_initializer=glorot_init, use_bias=True)(shortcut)
        shortcut = resample_op(shortcut)

    ### CONV PATH
    convpath = x
    if not is_first:
        convpath = norm(axis=feature_axis)(convpath)
        convpath = Activation('relu')(convpath)
    if resample == "UP":
        convpath = resample_op(convpath)

    convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                                      use_bias=True, padding='same')(convpath)

    convpath = norm(axis=feature_axis)(convpath)
    convpath = Activation('relu')(convpath)

    merge_layers = []

    if cond_bottleneck_layer is not None:
        cond_bottleneck_path = cond_bottleneck_layer(number_of_classes=number_of_classes, filters=nfilters,
                                     kernel_initializer=he_init)([convpath, cls])
        cond_bottleneck_path = norm(axis=feature_axis)(cond_bottleneck_path)
        cond_bottleneck_path = Activation('relu')(cond_bottleneck_path)
        merge_layers.append(cond_bottleneck_path)

    if uncond_bottleneck_layer is not None:
        uncond_bottleneck_path = cond_bottleneck_layer(kernel_size=(1, 1), filters=nfilters,
                                                        kernel_initializer=he_init)([convpath, cls])
        uncond_bottleneck_path = norm(axis=feature_axis)(uncond_bottleneck_path)
        uncond_bottleneck_path = Activation('relu')(uncond_bottleneck_path)
        merge_layers.append(uncond_bottleneck_path)

    if len(merge_layers) == 2:
        convpath = Add()(merge_layers)
    elif len(merge_layers) == 1:
        convpath = merge_layers[0]

    convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, kernel_initializer=he_init,
                          use_bias=True, padding='same')(convpath)

    if resample == "DOWN":
        convpath = resample_op(convpath)

    y = Add() ([shortcut, convpath])

    return y


def test_conditional_dense():
    from keras.models import Model, Input
    import numpy as np
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input((2,))
    cls = Input((1, ), dtype='int32')
    dence = ConditionalDense(number_of_classes=3, units=2, use_bias=True,
                            kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    rs_inp = Reshape((1, 1, 2))([inp])
    cv_sep = ConditionalConv2D(number_of_classes=3, kernel_size=(1, 1), filters=2, padding='valid', use_bias=True,
                               kernel_initializer=kernel_init, bias_initializer=kernel_init)([rs_inp, cls])
    m = Model([inp, cls], [dence, cv_sep])
    x = np.arange(2 * 2).reshape((2, 2))
    cls = np.expand_dims(np.arange(2) % 3, axis=-1)
    out1, out2 = m.predict([x, cls])
    out2 = np.squeeze(out2, axis=(1, 2))

    assert np.sum(np.abs(out1 - out2)) < 1e-5


def test_conditional_conv11():
    from keras.models import Model, Input
    import numpy as np
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input(batch_shape = (10, 10, 10, 10))
    cls = Input(batch_shape = (10, 1), dtype='int32')
    cv11 = ConditionalConv11(number_of_classes=3, filters=20,
                                            kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    cv_sep = ConditionalConv2D(number_of_classes=3, kernel_size=(1, 1), filters=20, padding='valid', use_bias=True,
                               kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    m = Model([inp, cls], [cv11, cv_sep])
    x = np.arange(10 * 1000).reshape((10, 10, 10, 10))
    cls = np.expand_dims(np.arange(10) % 3, axis=-1)
    out1, out2 = m.predict([x, cls])

    assert np.sum(np.abs(out1 - out2)) < 1e-5


def test_conditional_instance():
    from keras.models import Model, Input
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalInstanceNormalization(3, axis=-1, gamma_initializer=beta_init,
                                                           beta_initializer=beta_init)([inp, cls]))
    x = np.ones((3, 2, 2, 1))
    cls = np.expand_dims(np.arange(3), axis=-1)
    out = m.predict([x, cls])

    assert np.all(out[0] == 1)
    assert np.all(out[1] == 2)
    assert np.all(out[2] == 3)


def test_conditional_conv():
    from keras.models import Model, Input
    import numpy as np
    def kernel_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a

    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalConv2D(number_of_classes=3, filters=1,
             kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls]))
    x = np.ones((3, 2, 2, 1))
    cls = np.expand_dims(np.arange(3), axis=-1)
    cls[2] = 0
    out = m.predict([x, cls])

    assert np.all(out[0] == 5)
    assert np.all(out[1] == 10)
    assert np.all(out[2] == 5)

if __name__ == "__main__":
    #test_conditional_conv()
    #test_conditional_instance()
    #test_conditional_conv11()
    test_conditional_dense()
