from tf_imports import Model
from tf_imports import Input, Dropout, Flatten, Dense, Reshape, GaussianNoise
from tf_imports import Conv2D, Conv2DTranspose, MaxPooling2D
from models.base_model import BaseModel
from pathlib import Path
import const


class CAE(BaseModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def _create_model(self, input_shape):
        # encoder
        kernel_initializer = 'truncated_normal'
        # kernel_initializer = 'glorot_uniform'
        activation1 = 'relu'
        activation2 = 'tanh'

        encoder_inputs = Input(shape=input_shape, name='encoder_input')  # (N, M, 1)
        encoder = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_inputs)  # (N, M, 1)
        encoder = Conv2D(16, 3, strides=2, padding='same', activation=activation1, kernel_initializer=kernel_initializer)(encoder)  # (N/2, M/2, 16)
        encoder = Conv2D(32, 3, strides=2, padding='same', activation=activation1, kernel_initializer=kernel_initializer)(encoder)  # (N/4, M/4, 32)
        encoder = Conv2D(64, 3, strides=2, padding='same', activation=activation1, kernel_initializer=kernel_initializer)(encoder)  # (N/8, M/8, 64)
        encoder = Conv2D(128, 3, strides=2, padding='same', activation=activation1, kernel_initializer=kernel_initializer)(encoder)  # (N/16 , M/16, 128)
        encoder = Conv2D(256, 3, strides=2, padding='same', activation=activation1, kernel_initializer=kernel_initializer)(encoder)  # (N/32 , M/32, 256)
        encoder = Flatten()(encoder)  # (N/32 * M/32 * 256)
        encoder = Dropout(0.1)(encoder)  # (N/32 * M/32 * 256)
        encoder = Dense(1024, activation=activation2, kernel_initializer='truncated_normal')(encoder)  # (512)
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder, name="encoder")

        # decoder
        kernel_initializer = 'truncated_normal'
        # kernel_initializer = 'glorot_uniform'
        activation1 = 'relu'
        activation2 = 'relu'
        activation3 = 'sigmoid'

        decoder_inputs = Input(shape=encoder_model.output_shape[1:], name="decoder_input")  # (N)
        decoder = Dense(4096, activation=activation1, kernel_initializer=kernel_initializer)(decoder_inputs)  # (4096)
        decoder = Reshape((4, 4, 256))(decoder)  # (4, 4, 256)
        decoder = Conv2DTranspose(128, 3, strides=2, padding='same', activation=activation2, kernel_initializer=kernel_initializer)(decoder)  # (8, 8, 128)
        decoder = Conv2DTranspose(64, 3, strides=2, padding='same', activation=activation2, kernel_initializer=kernel_initializer)(decoder)  # (16, 16, 64)
        decoder = Conv2DTranspose(32, 3, strides=2, padding='same', activation=activation2, kernel_initializer=kernel_initializer)(decoder)  # (32, 32, 32)
        decoder = Conv2DTranspose(3, 3, strides=2, padding='same', activation=activation3, kernel_initializer=kernel_initializer)(decoder)  # (64, 64, 3)
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
