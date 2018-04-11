from keras.models import load_model
from keras.backend import tf as ktf
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.backend import function

class GAN(object):
    def __init__(self, generator, discriminator,
                 generator_optimizer=Adam(2e-4, beta_1=0, beta_2=0.9),
                 discriminator_optimizer=Adam(2e-4, beta_1=0, beta_2=0.9),
                 generator_adversarial_objective='ns-gan',
                 discriminator_adversarial_objective='ns-gan',
                 gradient_penalty_weight=10,
                 gradient_penalty_type='dragan',
                 additional_inputs_for_generator_train=[],
                 additional_inputs_for_discriminator_train=[],
                 custom_objects={},
                 lr_decay_schedule_generator=lambda iter: 1.0,
                 lr_decay_schedule_discriminator=lambda iter: 1.0,
                 **kwargs):

        assert generator_adversarial_objective in ['ns-gan', 'lsgan', 'wgan', 'hinge']
        assert discriminator_adversarial_objective in ['ns-gan', 'lsgan', 'wgan', 'hinge']
        assert gradient_penalty_type in ['dragan', 'wgan-gp']

        if type(generator) == str:
            self.generator = load_model(generator, custom_objects=custom_objects)
        else:
            self.generator = generator

        if type(discriminator) == str:
            self.discriminator = load_model(discriminator, custom_objects=custom_objects)
        else:
            self.discriminator = discriminator

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        generator_input = self.generator.input
        discriminator_input = self.discriminator.input

        if type(generator_input) == list:
            self.generator_input = generator_input
        else:
            self.generator_input = [generator_input]

        if type(discriminator_input) == list:
            self.discriminator_input = discriminator_input
        else:
            self.discriminator_input = [discriminator_input]

        self.generator_adversarial_objective = generator_adversarial_objective
        self.discriminator_adversarial_objective = discriminator_adversarial_objective

        self.compile_intermediate_variables()
        self.intermediate_variables_to_lists()
        self.additional_inputs_for_generator_train=additional_inputs_for_generator_train
        self.additional_inputs_for_discriminator_train=additional_inputs_for_discriminator_train
        self.gradient_penalty_weight = gradient_penalty_weight
        self.gradient_penalty_type = gradient_penalty_type

        self.lr_decay_schedule_generator = lr_decay_schedule_generator
        self.lr_decay_schedule_discriminator = lr_decay_schedule_discriminator

        self.generator_metric_names = []
        self.discriminator_metric_names = []

    def get_generator_adversarial_loss(self, loss_type):
        def ns_loss(logits):
            labels = ktf.ones_like(logits)
            return ktf.reduce_mean(ktf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        def ls_loss(logits):
            return ktf.reduce_mean((logits - 1) ** 2)

        def wgan(logits):
            return -ktf.reduce_mean(logits)

        def hinge(logits):
            return -ktf.reduce_mean(logits)

        losses = {'ns-gan': ns_loss(self.discriminator_fake_output[0]),
                  'lsgan': ls_loss(self.discriminator_fake_output[0]),
                  'wgan': wgan(self.discriminator_fake_output[0]),
                  'hinge': hinge(self.discriminator_fake_output[0])}
        self.generator_metric_names.append('fake')
        return losses[loss_type]

    def get_discriminator_adversarial_loss(self, loss_type):
        def ns_loss_true(logits):
            labels = ktf.ones_like(logits)
            return ktf.reduce_mean(ktf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        def ns_loss_fake(logits):
            labels = ktf.zeros_like(logits)
            return ktf.reduce_mean(ktf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        def ls_loss_true(logits):
            return ktf.reduce_mean((logits - 1) ** 2)

        def ls_loss_fake(logits):
            return ktf.reduce_mean(logits ** 2)

        def wgan_loss_true(logits):
            return -ktf.reduce_mean(logits)

        def wgan_loss_fake(logits):
            return ktf.reduce_mean(logits)

        def hinge_loss_true(logits):
            return ktf.reduce_mean(ktf.maximum(0.0, 1.0 - logits))

        def hinge_loss_fake(logits):
            return ktf.reduce_mean(ktf.maximum(0.0, 1.0 + logits))

        losses = {'ns-gan': [ns_loss_true(self.discriminator_real_output[0]),
                             ns_loss_fake(self.discriminator_fake_output[0])],
                  'lsgan': [ls_loss_true(self.discriminator_real_output[0]),
                             ls_loss_fake(self.discriminator_fake_output[0])],
                  'wgan': [wgan_loss_true(self.discriminator_real_output[0]),
                             wgan_loss_fake(self.discriminator_fake_output[0])],
                  'hinge': [hinge_loss_true(self.discriminator_real_output[0]),
                             hinge_loss_fake(self.discriminator_fake_output[0])]}

        self.discriminator_metric_names.append('true')
        self.discriminator_metric_names.append('fake')
        return losses[loss_type]

    def get_gradient_penalty_loss(self):
        if self.gradient_penalty_weight == 0:
            return []

        if type(self.discriminator_input) == list:
            batch_size = ktf.shape(self.discriminator_input[0])[0]
            ranks = [len(inp.get_shape().as_list()) for inp in self.discriminator_input]
        else:
            batch_size = ktf.shape(self.discriminator_input)[0]
            ranks = [len(self.discriminator_input.get_shape().as_list())]

        def cast_all(values, reference_type_vals):
            return [ktf.cast(alpha, dtype=ref.dtype) for alpha, ref in zip(values, reference_type_vals)]

        def std_if_not_int(val):
            if val.dtype.is_integer:
                return 0
            else:
                return ktf.stop_gradient(K.std(val, keepdims=True))

        def point_for_gp_wgan():
            weights = ktf.random_uniform((batch_size, 1), minval=0, maxval=1)
            weights = [ktf.reshape(weights, (-1, ) + (1, ) * (rank - 1)) for rank in ranks]
            weights = cast_all(weights, self.discriminator_input)
            points = [(w * r) + ((1 - w) * f) for r, f, w in zip(self.discriminator_input, self.generator_output, weights)]
            return points

        def points_for_dragan():
            alphas = ktf.random_uniform((batch_size, 1), minval=0, maxval=1)
            alphas = [ktf.reshape(alphas, (-1, ) + (1, ) * (rank - 1)) for rank in ranks]
            alphas = cast_all(alphas, self.discriminator_input)
            fake = [ktf.random_uniform(ktf.shape(t), minval=0, maxval=1) * std_if_not_int(t) * 0.5
                       for t in self.discriminator_input]
            fake = cast_all(fake, self.discriminator_input)

            points = [(w * r) + ((1 - w) * f) for r, f, w in zip(self.discriminator_input, fake, alphas)]
            return points

        points = {'wgan-gp': point_for_gp_wgan(), 'dragan': points_for_dragan()}
        points = points[self.gradient_penalty_type]

        gp_list = []
        disc_out = self.discriminator(points)
        if type(disc_out) != list:
            disc_out = [disc_out]
        gradients = ktf.gradients(disc_out[0], points)

        for gradient in gradients:
            if gradient is None:
                continue
            gradient = ktf.reshape(gradient, (batch_size, -1))
            gradient_l2_norm = ktf.sqrt(ktf.reduce_sum(ktf.square(gradient), axis=1))
            gradient_penalty = self.gradient_penalty_weight * ktf.square(1 - gradient_l2_norm)
            gp_list.append(ktf.reduce_mean(gradient_penalty))

        for i in range(len(gp_list)):
            self.discriminator_metric_names.append('gp_loss_' + str(i))
        return gp_list

    def compile_intermediate_variables(self):
        self.generator_output = self.generator(self.generator_input)
        self.discriminator_fake_output = self.discriminator(self.generator_output)
        self.discriminator_real_output = self.discriminator(self.discriminator_input)


    def intermediate_variables_to_lists(self):
        if type(self.generator_output) != list:
            self.generator_output = [self.generator_output]
        if type(self.discriminator_fake_output) != list:
            self.discriminator_fake_output = [self.discriminator_fake_output]
        if type(self.discriminator_real_output) != list:
            self.discriminator_real_output = [self.discriminator_real_output]

    def additional_generator_losses(self):
        return []

    def additional_discriminator_losses(self):
        return []

    def collect_updates(self, model):
        updates = []
        for l in model.layers:
            updates += l.updates
        return updates

    def compile_generator_train_op(self):
        loss_list = []
        adversarial_loss = self.get_generator_adversarial_loss(self.generator_adversarial_objective)
        loss_list.append(adversarial_loss)

        loss_list += self.additional_generator_losses()
        self.generator_loss_list = loss_list

        updates = []

        updates += self.collect_updates(self.discriminator)
        updates += self.collect_updates(self.generator)
        print (updates) 
        updates += self.generator_optimizer.get_updates(params=self.generator.trainable_weights, loss=sum(loss_list))

        lr_update = (self.lr_decay_schedule_generator(self.generator_optimizer.iterations) *
                                K.get_value(self.generator_optimizer.lr))
        updates.append(K.update(self.generator_optimizer.lr, lr_update))

        train_op = function(self.generator_input + self.additional_inputs_for_generator_train + [K.learning_phase()],
                            [sum(loss_list)] + loss_list, updates=updates)
        return train_op

    def compile_discriminator_train_op(self):
        loss_list = []
        adversarial_loss = self.get_discriminator_adversarial_loss(self.generator_adversarial_objective)
        loss_list += adversarial_loss
        loss_list += self.get_gradient_penalty_loss()
        loss_list += self.additional_discriminator_losses()

        updates = []

        updates += self.collect_updates(self.discriminator)
        updates += self.collect_updates(self.generator)

        print (updates)
        updates += self.discriminator_optimizer.get_updates(params=self.discriminator.trainable_weights, loss=sum(loss_list))
              

        inputs = self.discriminator_input + self.additional_inputs_for_discriminator_train +\
                 self.generator_input + self.additional_inputs_for_generator_train

        lr_update = (self.lr_decay_schedule_discriminator(self.discriminator_optimizer.iterations) *
                                K.get_value(self.discriminator_optimizer.lr))
        updates.append(K.update(self.discriminator_optimizer.lr, lr_update))

        train_op = function(inputs + [K.learning_phase()], [sum(loss_list)] + loss_list, updates=updates)
        return train_op

    def compile_generate_op(self):
        return function(self.generator_input + self.additional_inputs_for_generator_train + [K.learning_phase()], self.generator_output)

    def compile_validate_op(self):
        return function(self.generator_input + self.additional_inputs_for_generator_train + [K.learning_phase()],
                        [sum(self.generator_loss_list)] + self.generator_loss_list)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_losses_as_string(self, generator_losses, discriminator_losses):
        def combine(name_list, losses):
            losses = np.array(losses)
            if len(losses.shape) == 0:
                losses = losses.reshape((1, ))
            return '; '.join([name + ' = ' + str(loss) for name, loss in zip(name_list, losses)])
        generator_loss_str = combine(['Generator loss'] + self.generator_metric_names, generator_losses)
        discriminator_loss_str = combine(['Disciminator loss'] + self.discriminator_metric_names, discriminator_losses)
        return generator_loss_str, discriminator_loss_str
