import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) > version.parse("1.10.0"):
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.models import Model
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras import optimizers, losses
else:
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.models import Model
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    # noinspection PyUnresolvedReferences
    from keras.callbacks import EarlyStopping
    # noinspection PyUnresolvedReferences
    from tensorflow.keras import optimizers, losses