import os
import argparse
import const
import models
import utils
import numpy as np
import itertools
import traceback
from sklearn.model_selection import train_test_split
from matplotlib import image as mpimg

from keras import optimizers, losses

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist")


def main():
    args = parser.parse_args()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(os.path.join(dataset_dir, "meta.json"))
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(os.path.join(dataset_dir, "word2vec-captions.bin")))

    data = []
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = os.path.join(dataset_dir, meta_entry["image"])
        img_array = mpimg.imread(img_file_path)

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            if word2vec_captions.shape[0] > const.INPUT_SHAPE[0]:
                raise Exception("input range exceded")

            padding = ((0, const.INPUT_SHAPE[0] - word2vec_captions.shape[0]), (0, 0))
            word2vec_captions = np.pad(word2vec_captions, padding, 'constant', constant_values=0)
            word2vec_captions = (word2vec_captions.astype("float32") + 1.) / 2.

            data.append((word2vec_captions, img_array))

    x, y = tuple(zip(*data))
    x, y = (np.expand_dims(x, 4), np.array(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    optimizers_options = [
        optimizers.Adam(),
        optimizers.Adadelta(),
        optimizers.Adagrad(),
        optimizers.Adamax(),
        optimizers.SGD()
    ]

    losses_options = [
        losses.binary_crossentropy,
        losses.categorical_crossentropy,
        losses.sparse_categorical_crossentropy,
        losses.categorical_hinge,
        losses.squared_hinge,
        losses.kullback_leibler_divergence,
        losses.mean_squared_error,
        losses.mean_absolute_error,
        losses.mean_squared_logarithmic_error,
        losses.mean_absolute_percentage_error
    ]

    batch_size_options = [32, 64, 128]

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.dataset)
    for optimizer, loss, batch_size in itertools.product(optimizers_options, losses_options, batch_size_options):
        try:
            model.train(x_train, y_train, x_test, y_test, optimizer=optimizer, loss=loss, batch_size=batch_size)
        except Exception as exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
