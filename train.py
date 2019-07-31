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
# parser.add_argument("-dataset", help="dataset name", default="mnist30k")
parser.add_argument("-dataset", help="dataset name", default="flowers")
parser.add_argument("-optimizer-index", help="optimizer index", type=int, default=0)
parser.add_argument("-loss-index", help="loss index", type=int, default=0)
parser.add_argument("-batch-size-index", help="batch size index", type=int, default=0)
parser.add_argument("-lr-schedule-fn-index", help="lr schedule fn index", type=int, default=0)
parser.add_argument("-use-tpu", help="use tpu", action="store_true")
parser.add_argument("-gpus", help="number of gpus to use tpu", type=int, default=None)

lr_schedule_params = [
    [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
]


def lr_schedule(epoch, schedule):
    if epoch == 0:
        return schedule[0]
    elif epoch < 10:
        return schedule[1]
    elif epoch < 30:
        return schedule[2]
    elif epoch < 60:
        return schedule[3]
    elif epoch < 100:
        return schedule[4]
    else:
        return schedule[5]


def main():
    optimizer_options = ([
        optimizers.Adam(clipnorm=10.),  # 0
    ])

    loss_options = ([
        losses.mean_squared_error,  # 0
    ])

    batch_size_options = ([
        512,  # 0
    ])

    lr_schedule_options = ([
        lambda epoch, lr: lr_schedule(epoch, lr_schedule_params[0]),  # 0
        lambda epoch, lr: lr_schedule(epoch, lr_schedule_params[1]),  # 1
    ])

    args = parser.parse_args()

    dataset_dir = const.DATASETS_PATH[args.dataset]
    dataset_meta = utils.json_utils.load(Path(dataset_dir, "meta.json"))
    dataset_word2vec_captions = np.array(utils.pickle_utils.load(Path(dataset_dir, "word2vec-captions.bin")))

    data = []
    max_input_shape = None
    skipped = 0
    for meta_index, meta_entry in enumerate(dataset_meta):
        img_file_path = Path(dataset_dir, meta_entry["image"])
        img_array = mpimg.imread(str(img_file_path))

        for word2vec_captions in dataset_word2vec_captions[meta_index]:
            word2vec_captions = np.array(word2vec_captions)
            # print("Input {}.".format(word2vec_captions.shape))

            if word2vec_captions.shape[0] == 0:
                skipped += 1
                continue

            if word2vec_captions.shape[0] > const.INPUT_SHAPE[0]:
                print("Input range exceded {}.".format(word2vec_captions.shape))
                if max_input_shape is None or max_input_shape[0] < word2vec_captions.shape[0]:
                    max_input_shape = word2vec_captions.shape
                word2vec_captions = word2vec_captions[:const.INPUT_SHAPE[0]]

            padding = ((0, const.INPUT_SHAPE[0] - word2vec_captions.shape[0]), (0, 0))
            # print("Padding {}.".format(padding))

            word2vec_captions = np.pad(word2vec_captions, padding, 'constant', constant_values=0)
            word2vec_captions = (word2vec_captions.astype("float32") + 1.) / 2.

            data.append((word2vec_captions, img_array))

    print("Max input shape {}".format(max_input_shape))
    print("Skipped {}".format(skipped))

    x, y = tuple(zip(*data[:]))
    x, y = (np.expand_dims(x, 4), np.array(y))

    model = models.models_dict[args.model](const.INPUT_SHAPE, args.use_tpu, args.gpus)
    optimizer = optimizer_options[args.optimizer_index]
    loss = loss_options[args.loss_index]
    batch_size = batch_size_options[args.batch_size_index]
    lr_schedule_fn = lr_schedule_options[args.lr_schedule_fn_index]

    desc = "{} {} {}".format(optimizer.__class__.__name__, loss.__name__, batch_size)
    print("\n\n" + desc)

    output_checkpoint_inputs = const.OUTPUT_CHECKPOINT_INPUTS[args.dataset]

    try:
        out_folder = "tmp/train/{}/{}".format(args.model, args.dataset)
        model.compile(optimizer, loss)
        model.train(
            x=x,
            y=y,
            batch_size=batch_size,
            out_folder=out_folder,
            lr_schedule=lr_schedule_fn,
            output_checkpoint_inputs=output_checkpoint_inputs)
        # output_checkpoint_inputs = None)
    except Exception:
        print("ERROR!!")
        traceback.print_exc()


if __name__ == "__main__":
    main()
