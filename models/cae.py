from tf_imports import Model
from tf_imports import Input, Dropout, Flatten, Dense, Reshape, GaussianNoise
from tf_imports import Conv2D, MaxPooling2D, Conv2DTranspose
from models.base_model import BaseModel
from pathlib import Path
import const


class CAE(BaseModel):
    def __init__(self, input_shape, use_tpu=False, gpus=None):
        super().__init__(input_shape, use_tpu, gpus)

    def _create_model(self, input_shape):
        # N, M, _ = input_shape

        # encoder
        encoder_inputs = Input(shape=input_shape, name='encoder_input')  # (N, M, 1)

        encoder = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_inputs)  # (N, M, 1)

        encoder = Conv2D(2, 3, padding='same', activation='relu')(encoder)  # (N, M, 2)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/2 , M/2, 2)

        encoder = Conv2D(4, 3, padding='same', activation='relu')(encoder)  # (N/2 , M/2, 4)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/4 , M/4, 4)

        encoder = Conv2D(8, 3, padding='same', activation='relu')(encoder)  # (N/4 , M/4, 8)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/8 , M/8, 8)

        encoder = Conv2D(16, 3, padding='same', activation='relu')(encoder)  # (N/8 , M/8, 16)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/16 , M/16, 16)

        encoder = Conv2D(32, 3, padding='same', activation='relu')(encoder)  # (N/16 , M/16, 32)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/32 , M/32, 32)

        encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)  # (N/32 , M/32, 64)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/64 , M/64, 64)

        encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)  # (N/64 , M/64, 128)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/128 , M/128, 128)

        encoder = Conv2D(256, 3, padding='same', activation='relu')(encoder)  # (N/128 , M/128, 256)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/256 , M/256, 256)

        encoder = Conv2D(512, 3, padding='same', activation='relu')(encoder)  # (N/256 , M/256, 512)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/512 , M/512, 512)

        encoder = Flatten()(encoder)  # (512)
        encoder = Dropout(0.1)(encoder)  # (512)
        encoder = Dense(512, activation='tanh')(encoder)  # (512)

        encoder = Model(inputs=encoder_inputs, outputs=encoder, name="encoder")

        # decoder
        decoder_inputs = Input(shape=encoder.output_shape[1:], name="decoder_input")

        decoder = Dense(4096, activation="elu")(decoder_inputs)
        decoder = Reshape((4, 4, 256))(decoder)  # (4, 4, 256)
        decoder = Conv2DTranspose(128, 3, strides=2, padding='same', activation='elu')(decoder)  # (8, 8, 128)
        decoder = Conv2DTranspose(64, 3, strides=2, padding='same', activation='elu')(decoder)  # (16, 16, 64)
        decoder = Conv2DTranspose(32, 3, strides=2, padding='same', activation='elu')(decoder)  # (32, 32, 32)
        decoder = Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(decoder)  # (64, 64, 3)

        decoder = Model(inputs=decoder_inputs, outputs=decoder, name='decoder')

        # CAE model
        cae_outputs = decoder(encoder(encoder_inputs))
        model = Model(inputs=encoder_inputs, outputs=cae_outputs, name='cae')

        self.model_encoder = encoder
        self.model_decoder = decoder

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
        self._plot_model(self.model_encoder, str(Path(save_to_dir, "cae_encoder.png")))
        self._plot_model(self.model_decoder, str(Path(save_to_dir, "cae_decoder.png")))
