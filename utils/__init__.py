import time
import numpy as np
from utils import json_utils, pickle_utils, plot_utils


def uid():
    return str(time.time()).replace(".", "").ljust(17, "0")


def process_w2v_inputs(word2vec_captions, input_shape):
    word2vec_captions_temp = []
    for index, word2vec_caption in enumerate(word2vec_captions):
        padding = ((0, input_shape[0] - len(word2vec_caption)), (0, 0))
        word2vec_caption = np.pad(word2vec_caption, padding, 'constant', constant_values=0)
        word2vec_caption = (word2vec_caption.astype("float32") + 1.) / 2.
        word2vec_captions_temp.append(word2vec_caption)

    word2vec_captions_temp = np.expand_dims(np.array(word2vec_captions_temp), 4)
    return word2vec_captions_temp
