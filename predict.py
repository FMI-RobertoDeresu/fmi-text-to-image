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


def main():
    args = parser.parse_args()
    captions = [
        "one one one",
        "two three five",
        "three six nine",
        "zero two eight",
    ]

    word2vec_captions = np.array(Word2Vec.get_instance().get_embeddings_remote(captions))
    word2vec_captions = utils.process_w2v_inputs(word2vec_captions, const.INPUT_SHAPE)

    save_path_template = Path(args.train_results_dir, "plots", "{}.png")
    save_path_template.parent.mkdir(parents=True, exist_ok=True)

    weights_paths = list(Path(args.train_results_dir, "weights").glob(pattern="*"))
    if args.weights == "last":
        weights_paths = weights_paths[-1:]

    model = models.models_dict[args.model](const.INPUT_SHAPE, False, None)

    for weights_path in weights_paths:
        model.load_weights(weights_path)
        images = model.predict(x_predict=word2vec_captions)

        desc = weights_path.stem
        save_path = str(save_path_template).format(desc)
        utils.plot_utils.plot_multiple_images(images, title=desc, labels=captions, save_path=save_path)


if __name__ == "__main__":
    main()
