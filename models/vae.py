import const
from models.base_model import BaseModel
from tf_imports import K, Model, Lambda
from tf_imports import Input, Flatten, Dense, Reshape, GaussianNoise
from tf_imports import Conv2D, Conv2DTranspose
from pathlib import Path


class VAE(BaseModel):
    def __init__(self, input_shape, use_tpu=False, gpus=None):
        super().__init__(input_shape, use_tpu, gpus)

    def _create_model(self, input_shape):
        latent_dim = 2

        # VAE model = encoder + decoder
        # build encoder model
        encoder_inputs = Input(shape=input_shape, name='encoder_input')  # (N, M, 1)
        encoder_inputs = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_inputs)

        encoder = Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)  # (N/2 , M/2, 32)
        encoder = Conv2D(64, 3, activation='relu', strides=2, padding='same')(encoder)  # (N/4 , M/4, 64)
        encoder = Conv2D(128, 3, activation='relu', strides=2, padding='same')(encoder)  # (N/8 , M/8, 128)
        encoder = Conv2D(256, 3, activation='relu', strides=2, padding='same')(encoder)  # (N/16 , M/16, 256)
        encoder = Conv2D(512, 3, activation='relu', strides=2, padding='same')(encoder)  # (N/32 , M/32, 512)

        # generate latent vector Q(z|X)
        encoder = Flatten()(encoder)  # (N/32 * M/32 * 512)
        encoder = Dense(16, activation='relu')(encoder)  # (16)
        z_mean = Dense(latent_dim, name='z_mean')(encoder)  # (2)
        z_log_var = Dense(latent_dim, name='z_log_var')(encoder)  # (2)
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])  # (2)

        encoder = Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name='encoder')

        # decoder
        decoder_inputs = Input(shape=(latent_dim,), name='z_sampling')  # (2)

        decoder = Dense(4096, activation='relu')(decoder_inputs)  # (4096)
        decoder = Reshape((4, 4, 256))(decoder)  # (4, 4, 256)
        decoder = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(decoder)  # (8, 8, 128)
        decoder = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(decoder)  # (16, 16, 64)
        decoder = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(decoder)  # (32, 32, 32)
        decoder = Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(decoder)  # (64, 64, 16)
        decoder = Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(decoder)  # (128, 128, 8)
        decoder = Conv2DTranspose(3, 3, strides=1, padding='same', activation='relu')(decoder)  # (128, 128, 3)

        decoder = Model(inputs=decoder_inputs, outputs=decoder, name='decoder')

        # VAE model
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        vae = Model(inputs=encoder_inputs, outputs=vae_outputs, name='vae')

        self.model_encoder = encoder
        self.model_decoder = decoder

        self.z_mean = z_mean
        self.z_log_var = z_log_var

        return vae

    def _compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self.optimizer = optimizer
        self.loss = loss

        self.model.compile(optimizer=optimizer, loss=self._loss, metrics=[])
        self.model_compiled = True

    def _loss(self, y_true, y_pred):
        reconstruction_loss = self.loss(K.flatten(y_true), K.flatten(y_pred))
        reconstruction_loss *= const.OUTPUT_IMAGE_SIZE[0] * const.OUTPUT_IMAGE_SIZE[1]

        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def plot_model(self, save_to_dir):
        self._plot_model(self.model, str(Path(save_to_dir, "vae.png")))
        self._plot_model(self.model_encoder, str(Path(save_to_dir, "vae_encoder.png")))
        self._plot_model(self.model_decoder, str(Path(save_to_dir, "vae_decoder.png")))

    # reparameterization trick (instead of sampling from Q(z|X), sample eps = N(0,I))
    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]

        epsilon = K.random_normal(shape=(batch, dim))
        z = z_mean + K.exp(0.5 * z_log_var) * epsilon

        return z
