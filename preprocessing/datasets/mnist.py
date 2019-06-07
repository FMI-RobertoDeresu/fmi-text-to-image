import tensorflow as tf
import numpy as np
import time
import utils
from PIL import Image
from pathlib import Path

dataset_path = Path("../../datasets/mnist/{}".format(int(time.time())))
dataset_path.mkdir(parents=True, exist_ok=True)

letter_name = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

n_samples = 10000
rnd_index_list = np.random.randint(60000, size=(n_samples, 3))
img_pad_with = 9 * (3 - 1)

meta = []
for i, index_list in enumerate(rnd_index_list):
    images = [np.array(x_train[x], dtype='float') for x in index_list]
    labels = [str(y_train[x]) for x in index_list]

    labels_names = [letter_name[x] for x in labels]
    label_name = " ".join(labels_names)

    image = np.concatenate(tuple(images), axis=1)
    image = np.pad(image, ((img_pad_with, img_pad_with), (0, 0)), mode="constant", constant_values=0)

    img = Image.fromarray(image)
    img = img.convert("RGB")
    img = img.resize((128, 128), Image.ANTIALIAS)

    image_file_name = "img_{}_{}.png".format(str((i+1)).rjust(4, "0"), "_".join(labels))
    image_file_path = Path(dataset_path, "images", image_file_name)
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(image_file_path))

    meta.append({
        "image": Path("images", image_file_name).as_posix(),
        "label": labels,
        "captions": [label_name]
    })

meta_file_path = Path(dataset_path, "meta.json")
meta_file_path.parent.mkdir(parents=True, exist_ok=True)
utils.json_utils.dump(meta, meta_file_path)
