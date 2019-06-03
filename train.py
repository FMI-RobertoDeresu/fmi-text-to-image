import os
import argparse
import const
import models
import utils
import traceback
import numpy as np
import pathlib
from tensorflow.contrib.keras import optimizers, losses
from sklearn.model_selection import train_test_split
from matplotlib import image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist10k")
parser.add_argument("-optimizer-index", help="optimizer index", type=int, default=0)
parser.add_argument("-loss-index", help="loss index", type=int, default=0)
parser.add_argument("-batch-size-index", help="batch size index", type=int, default=0)

parser.add_argument("-use-tpu", help="use tpu", action="store_true")

optimizer_options = ([
    optimizers.Adam(clipnorm=5.),  # 0
    optimizers.Adadelta(clipnorm=5.),  # 1
    optimizers.Adagrad(clipnorm=5.),  # 2
    optimizers.Adamax(clipnorm=5.),  # 3
    optimizers.SGD(clipnorm=5.)  # 4
])  # [0:1]

loss_options = ([
    losses.binary_crossentropy,  # 0
    losses.categorical_crossentropy,  # 1
    losses.categorical_hinge,  # 2
    losses.squared_hinge,  # 3
    losses.kullback_leibler_divergence,  # 4
    losses.mean_squared_error,  # 5
    losses.mean_absolute_error,  # 6
    losses.mean_squared_logarithmic_error,  # 7
    losses.mean_absolute_percentage_error  # 8
])  # [6:9]

batch_size_options = ([
    32,
    64,
    128
])  # [:]


def main():
    args = parser.parse_args()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(os.path.join(dataset_dir, "meta.json"))
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(os.path.join(dataset_dir, "word2vec-captions.bin")))

    data = []
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = str(pathlib.Path(os.path.join(dataset_dir, meta_entry["image"])))
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

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.dataset, args.use_tpu)
    optimizer = optimizer_options[args.optimizer_index]
    loss = loss_options[args.loss_index]
    batch_size = batch_size_options[args.batch_size_index]

    desc = "{} {} {}".format(optimizer.__class__.__name__, loss.__name__, batch_size)
    print("\n\n" + desc)

    try:
        model.train(x_train, y_train, x_test, y_test, optimizer=optimizer, loss=loss, batch_size=batch_size)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
