from tf_imports import Model
from tf_imports import Input, Dropout, Flatten, Dense, Reshape
from tf_imports import Conv2D, MaxPooling2D, Conv2DTranspose
from models.base_model import BaseModel


class CAE(BaseModel):
    def __init__(self, input_shape, use_tpu=False, gpus=None):
        super().__init__(input_shape, use_tpu, gpus)

    def _create_model(self, input_shape, use_tpu, gpus, print_model_summary):
        # N, M, _ = input_shape
        # input
        input_layer = Input(shape=input_shape)  # (N, M, 1)

        # encoder
        encoder = Conv2D(2, 3, padding='same', activation='relu')(input_layer)  # (N, M, 2)
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
        encoder = Dense(1024)(encoder)  # (1024)
        encoder = Dropout(0.1)(encoder)  # (1024)
        encoder = Dense(512)(encoder)  # (512)

        # decoder
        decoder = Reshape((1, 1, 512))(encoder)  # (1, 1, 512)
        decoder = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(decoder)  # (2, 2, 512)
        decoder = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(decoder)  # (4, 4, 256)
        decoder = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(decoder)  # (8, 8, 128)
        decoder = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(decoder)  # (16, 16, 64)
        decoder = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(decoder)  # (32, 32, 32)
        decoder = Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(decoder)  # (64, 64, 16)
        decoder = Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(decoder)  # (128, 128, 8)
        decoder = Conv2DTranspose(3, 3, strides=1, padding='same', activation='relu')(decoder)  # (128, 128, 3)

        # CAE model
        model = Model(input_layer, decoder)
        if print_model_summary:
            model.summary()

        return model

    def _compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model_compiled = True
