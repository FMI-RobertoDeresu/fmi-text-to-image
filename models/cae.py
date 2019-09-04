from tf_imports import tf, Model, Input, Flatten, Reshape, GaussianNoise
from models.base_model import BaseModel
from models.layers import conv, deconv, dense, dropout, batchnorm
from models import activations
from pathlib import Path
import const

configs = [
    # [[64, 64, 64, 0, 0, 512, 0], [4 * 64, 64, 64, 64, 64]],  # 0,
    # [[64, 64, 64, 0, 0, 512, 256], [4 * 64, 64, 64, 64, 64]],  # 1,
    #
    # [[64, 64, 64, 64, 0, 512, 0], [4 * 64, 64, 64, 64, 64]],  # 2,
    # [[64, 64, 64, 64, 0, 512, 256], [4 * 64, 64, 64, 64, 64]],  # 3,
    #
    # [[128, 64, 32, 32, 0, 512, 0], [4 * 64, 64, 64, 64, 64]],  # 4,
    # [[128, 64, 32, 32, 0, 512, 256], [4 * 64, 64, 64, 64, 64]],  # 5,

    [[64, 64, 64, 64, 64, 512, 0], [4 * 64, 64, 64, 64, 64]],  # 6,
    [[64, 64, 64, 64, 64, 512, 256], [4 * 64, 64, 64, 64, 64]],  # 7,
]


class CAE(BaseModel):
    def __init__(self, input_shape, config_index):
        self.cfg_idx = config_index
        super().__init__(input_shape)

    def _create_model(self, input_shape):
        droprate = 0.20
        enc_cfg, dec_cfg = configs[self.cfg_idx]

        # ================== encoder ==================
        with tf.name_scope('encoder_input'):  # (N, M, 1)
            encoder_input = Input(shape=input_shape, name="encoder_input")

        with tf.name_scope('encoder_input_noise'):  # (N, M, 1)
            encoder = GaussianNoise(stddev=const.NOISE_STDDEV)(encoder_input)

        with tf.name_scope('encoder_conv_1'):  # (N/2, M/2, 32)
            encoder = conv(enc_cfg[0], strides=2)(encoder)
            encoder = batchnorm()(encoder)
            encoder = dropout(droprate)(encoder)

        with tf.name_scope('encoder_conv_2'):  # (N/4, M/4, 32)
            encoder = conv(enc_cfg[1], strides=2)(encoder)
            encoder = batchnorm()(encoder)
            encoder = dropout(droprate)(encoder)

        with tf.name_scope('encoder_conv_3'):  # (N/8, M/8, 32)
            encoder = conv(enc_cfg[2], strides=2)(encoder)
            encoder = batchnorm()(encoder)
            encoder = dropout(droprate)(encoder)

        if enc_cfg[3] > 0:
            with tf.name_scope('encoder_conv_4'):  # (N/10, M/10, 32)
                encoder = conv(enc_cfg[3], strides=2)(encoder)
                encoder = batchnorm()(encoder)
                encoder = dropout(droprate)(encoder)

        if enc_cfg[4] > 0:
            with tf.name_scope('encoder_conv_4'):  # (N/10, M/10, 32)
                encoder = conv(enc_cfg[4], strides=2)(encoder)
                encoder = batchnorm()(encoder)
                encoder = dropout(droprate)(encoder)

        with tf.name_scope('encoder_fully_connected_1'):  # (512)
            encoder = Flatten()(encoder)
            encoder = dense(enc_cfg[5], activation=activations.lrelu)(encoder)
            encoder = batchnorm()(encoder)
            encoder = dropout(droprate)(encoder)

        if enc_cfg[6] > 0:
            with tf.name_scope('encoder_fully_connected_2'):  # (256)
                encoder = dense(enc_cfg[6], activation=activations.relu)(encoder)

        # encoder_model = Model(inputs=encoder_input, outputs=encoder, name="encoder")

        # ================== decoder ==================
        with tf.name_scope('decoder_input'):  # (N)
            # decoder_inputs = Input(shape=encoder_model.output_shape[1:], name="decoder_input")
            decoder_inputs = encoder

        with tf.name_scope('decoder_fully_connected_1'):  # (256)
            decoder = dense(dec_cfg[0])(decoder_inputs)

        with tf.name_scope('decoder_reshape_1'):  # (2, 2, 64)
            decoder = Reshape((2, 2, dec_cfg[0] // 4))(decoder)

        with tf.name_scope('decoder_deconv_1'):  # (4, 4, 32)
            decoder = deconv(dec_cfg[1], strides=2)(decoder)
            decoder = deconv(dec_cfg[1], strides=1)(decoder)
            decoder = batchnorm()(decoder)
            decoder = dropout(droprate)(decoder)

        with tf.name_scope('decoder_deconv_2'):  # (8, 8, 32)
            decoder = deconv(dec_cfg[2], strides=2)(decoder)
            decoder = deconv(dec_cfg[2], strides=1)(decoder)
            decoder = batchnorm()(decoder)
            decoder = dropout(droprate)(decoder)

        with tf.name_scope('decoder_deconv_3'):  # (16, 16, 32)
            decoder = deconv(dec_cfg[3], strides=2)(decoder)
            decoder = deconv(dec_cfg[3], strides=1)(decoder)
            decoder = batchnorm()(decoder)
            decoder = dropout(droprate)(decoder)

        with tf.name_scope('decoder_deconv_4'):  # (32, 32, 32)
            decoder = deconv(dec_cfg[4], strides=2)(decoder)
            decoder = deconv(dec_cfg[4], strides=1)(decoder)
            decoder = batchnorm()(decoder)
            decoder = dropout(droprate)(decoder)

        with tf.name_scope('decoder_deconv_5'):  # (64, 64, 3)
            decoder = deconv(3, strides=2, activation=activations.relu)(decoder)

        # decoder_model = Model(inputs=decoder_inputs, outputs=decoder, name='decoder')

        # ================== CAE ==================
        model = Model(inputs=encoder_input, outputs=decoder, name='cae')

        # self.encoder_model = encoder_model
        # self.decoder_model = decoder_model

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
