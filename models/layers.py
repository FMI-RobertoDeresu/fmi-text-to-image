from tf_imports import Conv2D, Conv2DTranspose, Dropout, Dense


def conv(filters, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"):
    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer)


def deconv(filters, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"):
    return Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer)


def dense(units, activation="relu", kernel_initializer="he_normal"):
    return Dense(units, activation=activation, kernel_initializer=kernel_initializer)


def dropout(rate=0.15):
    return Dropout(rate=rate)
