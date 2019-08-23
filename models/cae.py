from tf_imports import Model
from tf_imports import Input, Flatten, Reshape, GaussianNoise
from models.base_model import BaseModel
from models.layers import conv, deconv, dense, dropout, batchnorm
from models import activations
from pathlib import Path
import const


class CAE(BaseModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def _create_model(self, input_shape):
        droprate = 0.20

        # ================== encoder ==================
        # inputs
        encoder_inputs = Input(shape=input_shape, name='encoder_input')  # (N, M, 1)
        encoder = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_inputs)  # (N, M, 1)

        # conv #1 => (N/2, M/2, 32)
        encoder = conv(32, strides=2)(encoder)
        encoder = batchnorm()(encoder)
        encoder = dropout(droprate)(encoder)

        # conv #2 => (N/4, M/4, 32)
        encoder = conv(32, strides=2)(encoder)
        encoder = batchnorm()(encoder)
        encoder = dropout(droprate)(encoder)

        # conv #3 => (N/8, M/8, 32)
        encoder = conv(32, strides=2)(encoder)
        encoder = batchnorm()(encoder)
        encoder = dropout(droprate)(encoder)

        # fully connected #1 => (2048)
        encoder = Flatten()(encoder)
        encoder = dense(512, activation=activations.lrelu)(encoder)
        encoder = batchnorm()(encoder)
        encoder = dropout(droprate)(encoder)

        # fully connected #2 => (256)
        encoder = dense(256, activation=activations.relu)(encoder)

        # encoder model
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder, name="encoder")

        # ================== decoder ==================
        # inputs
        decoder_inputs = Input(shape=encoder_model.output_shape[1:], name="decoder_input")  # (N)

        # fully connected #1 => (2, 2, 64)
        decoder = dense(256)(decoder_inputs)
        decoder = Reshape((2, 2, 64))(decoder)

        # deconv #1 => (4, 4, 32)
        decoder = deconv(32)(decoder)
        decoder = batchnorm()(decoder)
        decoder = dropout(droprate)(decoder)

        # deconv #2 => (8, 8, 32)
        decoder = deconv(32)(decoder)
        decoder = batchnorm()(decoder)
        decoder = dropout(droprate)(decoder)

        # deconv #3 => (16, 16, 32)
        decoder = deconv(32)(decoder)
        decoder = batchnorm()(decoder)
        decoder = dropout(droprate)(decoder)

        # deconv #4 => (32, 32, 32)
        decoder = deconv(32)(decoder)
        decoder = batchnorm()(decoder)
        decoder = dropout(droprate)(decoder)

        # deconv #5 => (64, 64, 3)
        decoder = deconv(3, activation=activations.relu)(decoder)

        # decoder model
        decoder_model = Model(inputs=decoder_inputs, outputs=decoder, name='decoder')

        # ================== CAE ==================
        cae_outputs = decoder_model(encoder_model(encoder_inputs))
        model = Model(inputs=encoder_inputs, outputs=cae_outputs, name='cae')

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        return model

    def _compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[])
        self.model_compiled = True

    def plot_model(self, save_to_dir):
        self._plot_model(self.model, str(Path(save_to_dir, "cae.png")))
        self._plot_model(self.encoder_model, str(Path(save_to_dir, "cae_encoder.png")))
        self._plot_model(self.decoder_model, str(Path(save_to_dir, "cae_decoder.png")))
