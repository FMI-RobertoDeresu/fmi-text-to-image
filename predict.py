import argparse
import models
import numpy as np
import utils
import pathlib
import time
import const
from models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist10k")
parser.add_argument("-weights", help="all or last", default="all")
parser.add_argument("-word2vec", help="local or remote", default="remote")


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

    if args.word2vec == "remote":
        word2vec_captions = np.array(word2vec.get_embeddings_remote(captions))
    else:
        word2vec_captions = np.array(word2vec.get_embeddings(captions))

    save_path_template = "tmp/out/{}_{{}}.png".format(int(time.time()))
    pathlib.Path(save_path_template).parent.mkdir(parents=True, exist_ok=True)

    word2vec_captions_temp = []
    for index, word2vec_caption in enumerate(word2vec_captions):
        padding = ((0, const.INPUT_SHAPE[0] - len(word2vec_caption)), (0, 0))
        word2vec_caption = np.pad(word2vec_caption, padding, 'constant', constant_values=0)
        word2vec_caption = (word2vec_caption.astype("float32") + 1.) / 2.
        word2vec_captions_temp.append(word2vec_caption)

    word2vec_captions = np.expand_dims(np.array(word2vec_captions_temp), 4)

    model_out_folder = "tmp/train/cae/{}".format(args.dataset)
    model = models.models_dict[args.model](const.INPUT_SHAPE, model_out_folder, False)

    train_results = utils.json_utils.load(model.train_results_path)
    train_sessions = train_results["training_sessions"]

    if args.weights == "last":
        train_sessions = train_sessions[-1:]

    for train_session in train_sessions:
        model.load_weights(train_session["weights_path"])
        images = model.predict(x_predict=word2vec_captions)

        # for caption, img in list(zip(captions, images)):
        #     save_path = save_path_template.format(caption.replace(" ", "_"))
        #     mpimg.imsave(save_path, img)

        desc = train_session["description"]
        save_path = save_path_template.format(desc)
        utils.plot_utils.plot_multiple_images(images, title=desc, labels=captions, save_path=save_path)


if __name__ == "__main__":
    main()
