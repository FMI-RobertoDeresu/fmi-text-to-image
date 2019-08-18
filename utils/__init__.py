import time
import numpy as np
import nltk
from utils.stopwords import stopwords
from utils import json_utils, pickle_utils, plot_utils

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w{3,}")


def uid():
    return str(time.time()).replace(".", "").ljust(17, "0")


def prepare_caption_text_for_word2vec(caption):
    captions_words = tokenizer.tokenize(caption)
    captions_words = list(filter(lambda x: x not in stopwords, captions_words))
    captions_words = list(filter(lambda x: len(x) > 2, captions_words))
    caption = " ".join(captions_words)
    return caption, captions_words


def process_w2v_inputs(word2vec_captions, input_shape):
    word2vec_captions_temp = []
    for index, word2vec_caption in enumerate(word2vec_captions):
        padding = ((0, input_shape[0] - len(word2vec_caption)), (0, 0))
        word2vec_caption = np.pad(word2vec_caption, padding, 'constant', constant_values=0)
        word2vec_caption = (word2vec_caption.astype("float32") + 1.) / 2.
        word2vec_captions_temp.append(word2vec_caption)

    word2vec_captions_temp = np.expand_dims(np.array(word2vec_captions_temp), 4)
    return word2vec_captions_temp
