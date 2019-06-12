import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) > version.parse("1.10.0"):
    # noinspection PyUnresolvedReferences, PyPep8Naming
    from tensorflow.contrib.keras.api.keras import backend as K, optimizers, losses
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.layers import Dropout, Flatten, Dense, Reshape, Lambda
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.models import Model
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.api.keras.callbacks import Callback, TensorBoard, EarlyStopping, LearningRateScheduler
    # noinspection PyUnresolvedReferences
    from tensorflow.python.keras.utils import multi_gpu_model
    # noinspection PyUnresolvedReferences
    from tensorflow.python.summary import summary as tf_summary
else:
    # noinspection PyUnresolvedReferences
    from tensorflow.keras import K, optimizers, losses
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.layers import Dropout, Flatten, Dense, Reshape, Lambda
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.models import Model
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.callbacks import Callback, TensorBoard, LearningRateScheduler
    # noinspection PyUnresolvedReferences
    from tensorflow.python.keras.utils import multi_gpu_model
    # noinspection PyUnresolvedReferences
    from keras.callbacks import EarlyStopping
    # noinspection PyUnresolvedReferences
    from tensorflow.python.summary import summary as tf_summary
