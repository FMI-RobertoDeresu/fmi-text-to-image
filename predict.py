import argparse
import models
import numpy as np
import utils
import pathlib
import time
import const
import matplotlib.image as mpimg
from models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist")


def main():
    args = parser.parse_args()
    word2vec = Word2Vec()

    captions = [
        "one",
        "seven one",
        "one one one",
        "two three five",
        "three six nine",
        "zero two eight",
    ]

    word2vec_captions = np.array(word2vec.get_embeddings_remote(captions))

    save_path_template = "tmp/out/{}_{{}}.jpg".format(int(time.time()))
    pathlib.Path(save_path_template).parent.mkdir(parents=True, exist_ok=True)

    word2vec_captions_temp = []
    for index, word2vec_caption in enumerate(word2vec_captions):
        padding = ((0, const.INPUT_SHAPE[0] - len(word2vec_caption)), (0, 0))
        word2vec_caption = np.pad(word2vec_caption, padding, 'constant', constant_values=0)
        word2vec_caption = (word2vec_caption.astype("float32") + 1.) / 2.
        word2vec_captions_temp.append(word2vec_caption)

    word2vec_captions = np.expand_dims(np.array(word2vec_captions_temp), 4)

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.dataset)
    model.load_weights()

    imgs = model.predict(x_predict=word2vec_captions)

    for caption, img in list(zip(captions, imgs)):
        save_path = save_path_template.format(caption.replace(" ", "_"))
        mpimg.imsave(save_path, img)

    utils.plot_utils.plot_multiple_images(imgs, title=model.identifier, labels=captions)


if __name__ == "__main__":
    main()