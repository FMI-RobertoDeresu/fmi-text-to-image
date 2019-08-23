from tf_imports import Conv2D, Conv2DTranspose, Dropout, Dense, MaxPooling2D, BatchNormalization
from models.activations import lrelu, sigmoid, selu, elu

kernel_init = "lecun_normal"
# kernel_initializer =  "he_normal"


def conv(filters, kernel_size=3, strides=1, padding="same", activation=lrelu, kernel_initializer=kernel_init):
    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer)


def deconv(filters, kernel_size=5, strides=2, padding="same", activation=lrelu, kernel_initializer=kernel_init):
    return Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        use_bias=False,
        kernel_initializer=kernel_initializer)


def dense(units, activation=sigmoid, kernel_initializer=kernel_init):
    return Dense(units, activation=activation, kernel_initializer=kernel_initializer)


def maxpool(pool_size=2, strides=2, padding="same"):
    return MaxPooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )


def batchnorm():
    return BatchNormalization()


def dropout(rate):
    return Dropout(rate=rate)
