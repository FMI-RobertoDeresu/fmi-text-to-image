import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) > version.parse("1.10.0"):
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.models import Model
    # noinspection PyUnresolvedReferences, PyPep8Naming
    from tensorflow.contrib.keras.api.keras import backend as K, optimizers, losses, activations
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, \
        Flatten, Dense, Reshape, Lambda, GaussianNoise, BatchNormalization, Activation
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.callbacks import Callback, TensorBoard, EarlyStopping, LearningRateScheduler
    # noinspection PyUnresolvedReferences
    from tensorflow.python.summary import summary as tf_summary
    # noinspection PyUnresolvedReferences
    from tensorflow.python.ops import array_ops, math_ops
elif version.parse(tf.__version__) > version.parse("1.8.0"):
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.models import Model
    # noinspection PyUnresolvedReferences
    from tensorflow.keras import backend as K, optimizers, losses, activations
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten, Dense, \
        Reshape, Lambda, GaussianNoise, BatchNormalization, Activation
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.callbacks import Callback, TensorBoard, LearningRateScheduler
    # noinspection PyUnresolvedReferences
    from keras.callbacks import EarlyStopping
    # noinspection PyUnresolvedReferences
    from tensorflow.python.summary import summary as tf_summary
    # noinspection PyUnresolvedReferences
    from tensorflow.python.ops import array_ops, math_ops
