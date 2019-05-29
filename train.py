import os
import argparse
import const
import models
import utils
import numpy as np
from matplotlib import image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist")


def main():
    args = parser.parse_args()
    model = models.models_dict[args.model]()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(os.path.join(dataset_dir, "meta.json"))
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(os.path.join(dataset_dir, "word2vec-captions.bin")))

    train_data = []
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = os.path.join(dataset_dir, meta_entry["image"])
        img_array = mpimg.imread(img_file_path)

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            if word2vec_captions.shape[0] > 30:
                raise Exception("input range exceded")

            padding = ((0, 30 - word2vec_captions.shape[0]), (0, 0))
            word2vec_captions = np.pad(word2vec_captions, padding, 'constant', constant_values=0)
            word2vec_captions = (word2vec_captions.astype("float32") + 1.) / 2.

            train_data.append((word2vec_captions, img_array))

    x_train, y_train = tuple(zip(*train_data))
    x_train = np.expand_dims(x_train, 4)
    y_train = np.array(y_train)

    model.train(x_train, y_train, args.dataset)


if __name__ == "__main__":
    main()
