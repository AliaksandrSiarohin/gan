from keras import backend as K
from keras.optimizers import Adam
from gan import GAN
from wgan_gp import gradient_peanalty

class GAN_GP(GAN):
    """
        Class for representing GAN_GP
    """
    def __init__(self, generator, discriminator,
                       gradient_penalty_weight = 10,
                       generator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9),
                       discriminator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9),
                       **kwargs):
        super(GAN_GP, self).__init__(generator, discriminator, generator_optimizer=generator_optimizer,
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
        _, metrics = super(GAN_GP, self)._compile_discriminator_loss()
        true_loss, fake_loss = metrics

        real, fake, discriminator = self.get_data_for_gradient_penalty()
        gp_fn_list = gradient_peanalty(real, fake, self._gradient_penalty_weight, discriminator)

        def gp_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), gp_fn_list), K.zeros((1, )))

        def discriminator_loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred) + gp_loss(y_true, y_pred)

        return discriminator_loss, gp_fn_list + [true_loss, fake_loss]
