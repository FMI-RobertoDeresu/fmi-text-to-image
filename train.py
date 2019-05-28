import os
import argparse
import const
import models
import utils
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model name", default="cae")
parser.add_argument("dataset", help="dataset name", default="mnist")

if __name__ == "__main__":
    args = parser.parse_args()
    model = models.models_dict[args.model]

    dataset_dir = os.path.join("datasets", const.DATASETS[args.dataset])
    dataset_meta = utils.json_utils.load(os.path.join(dataset_dir, "meta.json"))
    dataset_word2vec_captions = utils.pickle_utils.load(os.path.join(dataset_dir, "word2vec-captions.bin"))

    train_data = []
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = os.path.join(dataset_dir, meta_entry["image"])
        img = Image.open(img_file_path)
        img_array = np.asarray(img).astype("float32") / 255.

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            word2vec_captions_pad = np.pad(word2vec_captions, ((0, 30), (0, 0)), 'constant', constant_values=0)
            word2vec_captions_pad = (word2vec_captions_pad.astype("float32") + 1.) / 2.

            train_data.append((word2vec_captions_pad, img_array))

    model.train(train_data, args.dataset)
