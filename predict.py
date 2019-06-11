import argparse
import models
import numpy as np
import utils
import const
from pathlib import Path
from models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="vae")
parser.add_argument("-train-results-dir", help="results directory", default="tmp/train/vae/mnist30k")
parser.add_argument("-weights", help="all or last", default="all")
parser.add_argument("-word2vec", help="local or remote", default="remote")


def main():
    args = parser.parse_args()
    word2vec = Word2Vec()

    captions = [
        "one one one",
        "two three five",
        "three six nine",
        "zero two eight",
    ]

    if args.word2vec == "remote":
        word2vec_captions = np.array(word2vec.get_embeddings_remote(captions))
    else:
        word2vec_captions = np.array(word2vec.get_embeddings(captions))

    word2vec_captions_temp = []
    for index, word2vec_caption in enumerate(word2vec_captions):
        padding = ((0, const.INPUT_SHAPE[0] - len(word2vec_caption)), (0, 0))
        word2vec_caption = np.pad(word2vec_caption, padding, 'constant', constant_values=0)
        word2vec_caption = (word2vec_caption.astype("float32") + 1.) / 2.
        word2vec_captions_temp.append(word2vec_caption)

    word2vec_captions = np.expand_dims(np.array(word2vec_captions_temp), 4)

    results_file_path = Path(args.train_results_dir, "results.json")
    train_results = utils.json_utils.load(results_file_path)
    train_sessions = train_results["training_sessions"]

    save_path_template = Path(args.train_results_dir, "plots", "{}.png")
    save_path_template.parent.mkdir(parents=True, exist_ok=True)

    if args.weights == "last":
        train_sessions = train_sessions[-1:]

    model = models.models_dict[args.model](const.INPUT_SHAPE, False, None)

    for train_session in train_sessions:
        weights_path = Path(args.train_results_dir, train_session["weights_path"])
        model.load_weights(weights_path)
        images1 = model.predict(x_predict=word2vec_captions)

        desc = train_session["description"]
        save_path = str(save_path_template).format(desc)
        utils.plot_utils.plot_multiple_images(images1, title=desc, labels=captions, save_path=save_path)


if __name__ == "__main__":
    main()
