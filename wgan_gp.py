from keras import backend as K
from keras.optimizers import Adam
from wgan import WGAN
from functools import partial


def gradient_peanalty(real, fake, gp_weight, discriminator):
    if type(real) == list:
        batch_size = K.shape(real[0])[0:1]
        input_shape = K.int_shape(real[0])
    else:
        batch_size = K.shape(real)[0:1]
        input_shape = K.int_shape(real)

    weights = K.random_uniform(batch_size)

    weights = K.reshape(weights, (-1, ) + (1, ) * (len(input_shape) - 1))
    averaged_samples = [(weights * r) + ((1 - weights) * f) for r, f in zip(real, fake)]

    gp_list = []
    gradients = K.gradients(K.sum(discriminator(averaged_samples)), averaged_samples)
    for gradient in gradients:
        gradient = K.reshape(gradient, (batch_size[0], -1))
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradient), axis=1))
        gradient_penalty = gp_weight * K.square(1 - gradient_l2_norm)
        gp_list.append(K.mean(gradient_penalty))

    fn = lambda y_true, y_pred, index: gp_list[index]
    gp_fn_list = [partial(fn, index=i) for i in range(len(gp_list))]
    for i, gp_fn in enumerate(gp_fn_list):
        gp_fn.__name__ = 'gp_loss_' + str(i)
    return gp_fn_list

class WGAN_GP(WGAN):
    """
        Class for representing WGAN_GP (https://arxiv.org/abs/1704.00028)
    """
    def __init__(self, generator, discriminator,
                       gradient_penalty_weight=10,
                       generator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9),
                       discriminator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9),
                       **kwargs):
        super(WGAN, self).__init__(generator, discriminator, generator_optimizer=generator_optimizer,
                                      discriminator_optimizer = discriminator_optimizer, **kwargs)
        self._gradient_penalty_weight = gradient_penalty_weight
        
        self.generator_metric_names = []
        self.discriminator_metric_names = ['gp_loss_' + str(i) for i in range(len(self._discriminator_input))] + ['true', 'fake']

    def get_data_for_gradient_penalty(self):
        real = self._discriminator_input
        fake = self._discriminator_fake_input
        discriminator = self._discriminator

        return real, fake, discriminator

    def _compile_discriminator_loss(self):        
        _, metrics = super(WGAN_GP, self)._compile_discriminator_loss()
        true_loss, fake_loss = metrics

        real, fake, discriminator = self.get_data_for_gradient_penalty()
        gp_fn_list = gradient_peanalty(real, fake, self._gradient_penalty_weight, discriminator)

        def gp_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), gp_fn_list), K.zeros((1, )))
            
        def discriminator_wasserstein_loss(y_true, y_pred):           
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred) + gp_loss(y_true, y_pred)
        
        return discriminator_wasserstein_loss, gp_fn_list + [true_loss, fake_loss]
