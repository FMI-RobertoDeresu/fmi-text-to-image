import argparse
import const
import models
import utils
import traceback
import numpy as np
from pathlib import Path
from matplotlib import image as mpimg
from tf_imports import optimizers, losses, K, tf
from models.word2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
# parser.add_argument("-model", help="model name", default="vae")
parser.add_argument("-dataset", help="dataset name", default="mnist1k")
# parser.add_argument("-dataset", help="dataset name", default="mnist30k")
# parser.add_argument("-dataset", help="dataset name", default="flowers")
parser.add_argument("-optimizer-index", help="optimizer index", type=int, default=0)
parser.add_argument("-loss-index", help="loss index", type=int, default=0)
parser.add_argument("-batch-size-index", help="batch size index", type=int, default=0)
parser.add_argument("-cfg-index", help="cfg index", type=int, default=0)


def main():
    optimizer_options = ([
        optimizers.Adam(lr=0.001),  # 0
    ])

    loss_options = ([
        lambda y_true, y_pred: tf.constant(100.0) * losses.mean_squared_error(K.flatten(y_true), K.flatten(y_pred)),
        lambda y_true, y_pred: tf.constant(100.0) * losses.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)),
    ])

    batch_size_options = ([
        64,  # 1
    ])

    args = parser.parse_args()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(Path(dataset_dir, "meta.json"))  # [:10000]
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(Path(dataset_dir, "word2vec-captions.bin")))

    data = []
    max_input_shape = None
    skipped = 0
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = Path(dataset_dir, meta_entry["image"])
        img_array = mpimg.imread(str(img_file_path))

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            word2vec_captions = np.array(word2vec_captions)
            if word2vec_captions.shape[0] == 0:
                skipped += 1
                continue

            if max_input_shape is None or max_input_shape[0] < word2vec_captions.shape[0]:
                max_input_shape = word2vec_captions.shape

            if word2vec_captions.shape[0] > const.INPUT_SHAPE[0]:
                print("Input range exceded {}.".format(word2vec_captions.shape))
                word2vec_captions = word2vec_captions[:const.INPUT_SHAPE[0]]

            padding = ((0, const.INPUT_SHAPE[0] - word2vec_captions.shape[0]), (0, 0))
            word2vec_captions = np.pad(word2vec_captions, padding, 'constant', constant_values=0)
            word2vec_captions = (word2vec_captions.astype("float32") + 1.) / 2.

            data.append((word2vec_captions, img_array))

    print("Max input shape {}".format(max_input_shape))
    print("Skipped {}".format(skipped))

    x, y = tuple(zip(*data[:]))
    x, y = (np.expand_dims(x, 4), np.array(y))

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.cfg_index)
    optimizer = optimizer_options[args.optimizer_index]
    loss = loss_options[args.loss_index]
    batch_size = batch_size_options[args.batch_size_index]

    test_captions = const.OUTPUT_CHECKPOINT_INPUTS[args.dataset]
    test_captions_word2vec = None
    if test_captions is not None:
        test_captions = [utils.prepare_caption_text_for_word2vec(x)[0] for x in test_captions]
        test_captions_word2vec = np.array(Word2Vec.get_instance().get_embeddings_remote(test_captions))
        test_captions_word2vec = utils.process_w2v_inputs(test_captions_word2vec, const.INPUT_SHAPE)

    try:
        # path = Path("tmp/plot")
        # path.mkdir(exist_ok=True)
        # model.plot_model(str(path))
        # return

        out_folder = "tmp/train/{}/{}".format(args.model, args.dataset)
        model.compile(optimizer, loss)
        model.train(
            x=x,
            y=y,
            batch_size=batch_size,
            out_folder=out_folder,
            output_checkpoint_inputs_word2vec=test_captions_word2vec)
    except Exception:
        print("ERROR!!")
        traceback.print_exc()


if __name__ == "__main__":
    main()
