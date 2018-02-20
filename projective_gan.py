from gan import GAN

class ProjectiveGAN(GAN):
    def compile_intermediate_variables(self):
        self.generator_output = [self.generator(self.generator_input), self.generator_input[1]]
        self.discriminator_fake_output = self.discriminator(self.generator_output)
        self.discriminator_real_output = self.discriminator(self.discriminator_input)
