import argparse
import const
import models
import utils
import traceback
import numpy as np
from pathlib import Path
from matplotlib import image as mpimg
from tf_imports import optimizers, losses

parser = argparse.ArgumentParser()
# parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-model", help="model name", default="vae")
# parser.add_argument("-dataset", help="dataset name", default="mnist1k")
parser.add_argument("-dataset", help="dataset name", default="mnist30k")
parser.add_argument("-optimizer-index", help="optimizer index", type=int, default=0)
parser.add_argument("-loss-index", help="loss index", type=int, default=0)
parser.add_argument("-batch-size-index", help="batch size index", type=int, default=0)
parser.add_argument("-use-tpu", help="use tpu", action="store_true")
parser.add_argument("-gpus", help="number of gpus to use tpu", type=int, default=None)


def lr_schedule(epoch, lr=None):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    optimizer_options = ([
        optimizers.Adam(lr=lr_schedule(0), clipnorm=0.001),  # 0
    ])

    loss_options = ([
        losses.mean_squared_error,  # 0
        losses.binary_crossentropy,  # 1
    ])

    batch_size_options = ([
        256,  # 0
    ])

    args = parser.parse_args()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(Path(dataset_dir, "meta.json"))
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(Path(dataset_dir, "word2vec-captions.bin")))

    data = []
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = Path(dataset_dir, meta_entry["image"])
        img_array = mpimg.imread(str(img_file_path))

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            if word2vec_captions.shape[0] > const.INPUT_SHAPE[0]:
                raise Exception("input range exceded")

            padding = ((0, const.INPUT_SHAPE[0] - word2vec_captions.shape[0]), (0, 0))
            word2vec_captions = np.pad(word2vec_captions, padding, 'constant', constant_values=0)
            word2vec_captions = (word2vec_captions.astype("float32") + 1.) / 2.

            data.append((word2vec_captions, img_array))

    x, y = tuple(zip(*data))
    x, y = (np.expand_dims(x, 4), np.array(y))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    # noise = np.random.normal(loc=0, scale=0.03, size=x_train.shape)
    # x_train += noise

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.use_tpu, args.gpus)
    optimizer = optimizer_options[args.optimizer_index]
    loss = loss_options[args.loss_index]
    batch_size = batch_size_options[args.batch_size_index]

    desc = "{} {} {}".format(optimizer.__class__.__name__, loss.__name__, batch_size)
    print("\n\n" + desc)

    output_checkpoint_inputs = [
        "one one one",
        "two three five",
        "three six nine",
        "zero two eight",
    ]

    try:
        out_folder = "tmp/train/{}/{}".format(args.model, args.dataset)
        model.compile(optimizer, loss)
        model.train(x, y, batch_size, out_folder, lr_schedule, output_checkpoint_inputs)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
