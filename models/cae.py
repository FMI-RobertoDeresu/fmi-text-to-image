from tf_imports import Model
from tf_imports import Input, Flatten, Reshape, GaussianNoise
from models.base_model import BaseModel
from models.layers import conv, deconv, dense, dropout
from pathlib import Path
import const


class CAE(BaseModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def _create_model(self, input_shape):
        # encoder

        encoder_inputs = Input(shape=input_shape, name='encoder_input')  # (N, M, 1)
        encoder = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_inputs)  # (N, M, 1)
        encoder = conv(16)(encoder)  # (N/2, M/2, 16)
        encoder = conv(16)(encoder)  # (N/4, M/4, 32)
        encoder = conv(32)(encoder)  # (N/8, M/8, 64)
        encoder = conv(32)(encoder)  # (N/16 , M/16, 128)
        encoder = conv(64)(encoder)  # (N/32 , M/32, 256)
        encoder = conv(64)(encoder)  # (N/64 , M/64, 512)
        encoder = conv(128)(encoder)  # (N/128 , M/128, 1024)
        encoder = conv(128)(encoder)  # (N/256 , M/128, 1024)
        encoder = conv(256)(encoder)  # (N/512 , M/128, 1024)
        encoder = Flatten()(encoder)  # (N/512 * M/512 * 256)
        encoder = dropout()(encoder)  # (N/32 * M/32 * 256)
        encoder = dense(1024, "sigmoid")(encoder)  # (512)
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder, name="encoder")

        # decoder
        decoder_inputs = Input(shape=encoder_model.output_shape[1:], name="decoder_input")  # (N)
        decoder = dense(4096)(decoder_inputs)  # (4096)
        decoder = Reshape((4, 4, 256))(decoder)  # (4, 4, 256)
        decoder = deconv(128)(decoder)  # (8, 8, 128)
        decoder = deconv(128, strides=1)(decoder)  # (8, 8, 128)
        decoder = deconv(64)(decoder)  # (16, 16, 64)
        decoder = deconv(64, strides=1)(decoder)  # (16, 16, 64)
        decoder = deconv(32)(decoder)  # (32, 32, 32)
        decoder = deconv(32, strides=1)(decoder)  # (32, 32, 32)
        decoder = deconv(3)(decoder)  # (64, 64, 3)
        decoder_model = Model(inputs=decoder_inputs, outputs=decoder, name='decoder')

        # CAE model
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
