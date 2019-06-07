import utils
import nltk
import argparse
import tensorflow as tf
import time
import re
import numpy as np
import skimage
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.python.client import device_lib

parser = argparse.ArgumentParser()
parser.add_argument("-action", help="action to execute", default="main")


def main():
    noise_image()


def captions_lengths():
    datasets_paths = Path("datasets").glob(pattern="*")
    for dataset_path in datasets_paths:
        meta_file_path = Path(dataset_path, "meta.json")
        meta = utils.json_utils.load(meta_file_path)

        max_length = 0
        for meta_entry in meta:
            for caption in meta_entry["captions"]:
                try:
                    words = re.findall(r"\w+", caption)
                    max_length = max(max_length, len(words))
                except:
                    print(meta_entry["image"])

        print("{} - {}".format(dataset_path.name, max_length))

def max_words_per_caption():
    datasets_names = ["mnist1k-3x", "oxford-102-flowers", "cub-200-2011", "flickr30k", "coco-train-2014"]
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w{3,}")
    stopwords = set(nltk.corpus.stopwords.words('english'))
    print(stopwords)

    for dataset_name in datasets_names[:]:
        meta_file_path = Path("datasets/{}".format(dataset_name), "meta.json")
        meta = utils.json_utils.load(meta_file_path)

        max_n_words = 0
        max_n_words_caption = 0
        max_n_words_image = ""
        for index, meta_entry in enumerate(meta):
            for caption in meta_entry["captions"]:
                words = tokenizer.tokenize(caption)
                words = list(filter(lambda x: x not in stopwords, words))
                if len(words) > max_n_words:
                    max_n_words = len(words)
                    max_n_words_caption = caption
                    max_n_words_image = meta_entry["image"]

        print("{}: {} ({} - {})".format(dataset_name, max_n_words, max_n_words_image, max_n_words_caption))


def using_gpu():
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    # Creates a session with log_device_placement set to True.
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(c))


def test_tpu_flops():
    n = 4096
    count = 100

    def flops():
        x = tf.random_uniform([n, n])
        y = tf.random_uniform([n, n])

        def _matmul(x, y):
            return tf.tensordot(x, y, axes=[[1], [0]]), y

        return tf.reduce_sum(tf.contrib.tpu.repeat(count, _matmul, [x, y]))

    tpu_ops = tf.contrib.tpu.batch_parallel(flops, [], num_shards=8)
    tpu_address = 'grpc://' + "10.240.1.2"

    session = tf.Session(tpu_address)
    try:
        print('Warming up...')
        session.run(tf.contrib.tpu.initialize_system())
        session.run(tpu_ops)
        print('Profiling')
        start = time.time()
        session.run(tpu_ops)
        end = time.time()
        elapsed = end - start
        print(elapsed, 'TFlops: {:.2f}'.format(1e-12 * 8 * count * 2 * n * n * n / elapsed))
    finally:
        session.run(tf.contrib.tpu.shutdown_system())
        session.close()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    for index, device in enumerate(local_device_protos):
        print("\nDevice {}:".format(index))
        print(device)


def replace():
    results_path = Path("./tmp/train/cae/mnist10k/results.json")
    train_results = utils.json_utils.load(results_path)

    train_sessions = train_results["training_sessions"]
    for train_session in train_sessions:
        train_session["weights_path"] = train_session["best_checkpoint_path"].format(
            out_folder=train_session["description"]) + ".data-00000-of-00001"
        train_session.pop("best_checkpoint_path")
        train_session.pop("early_stopping")

    train_results["training_sessions"] = train_sessions
    utils.json_utils.dump(train_results, results_path)


def noise_image():
    img_url = "https://i.guim.co.uk/img/media/4ddba561156645952502f7241bd1a64abd0e48a3/0_1251_3712_2225/master/" \
              "3712.jpg?width=1920&quality=85&auto=format&fit=max&s=1280341b186f8352416517fc997cd7da"
    img = skimage.io.imread(img_url) / 255.0

    def plot_noise(img, mode, r, c, i, var=0.01):
        plt.subplot(r, c, i)
        if mode is not None:
            # gimg = skimage.util.random_noise(img, mode=mode, var=var)
            gimg = np.random.normal(loc=0, scale=0.1, size=img.shape) + img
            plt.imshow(gimg)
        else:
            plt.imshow(img)
        plt.title(mode)
        plt.axis("off")

    plt.figure(figsize=(18, 24))
    r = 4
    c = 2
    plot_noise(img, "gaussian", r, c, 1, 0.01)
    # plot_noise(img, "localvar", r, c, 2)
    # plot_noise(img, "poisson", r, c, 3)
    # plot_noise(img, "salt", r, c, 4)
    # plot_noise(img, "pepper", r, c, 5)
    # plot_noise(img, "s&p", r, c, 6)
    # plot_noise(img, "speckle", r, c, 7)
    plot_noise(img, None, r, c, 8)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    actions_dict = {
        "main": main,
        "test_cae": test_cae,
        "max_words_per_caption": max_words_per_caption,
        "using_gpu": using_gpu,
        "test_tpu_flops": test_tpu_flops,
        "get_available_gpus": get_available_gpus
    }

    actions_dict[args.action]()
