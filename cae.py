import tensorflow as tf
import tflearn

class CAE:
    def __init__(self, input_data):

        input = tf.placeholder(shape=(None, 28, 28, 1))

        encoder = tflearn.layers.conv.conv_2d()
