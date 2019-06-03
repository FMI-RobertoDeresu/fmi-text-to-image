import pathlib
import os
import utils
import nltk
import argparse
import tensorflow as tf
import time
import subprocess
from tensorflow.python.client import device_lib

parser = argparse.ArgumentParser()
parser.add_argument("-action", help="action to execute", default="main")


def main():
    subprocess.call(["python", "train.py", "-model", "test1", "-dataset", "test2", "-batch-size-index", "3"])


def test_cae():
    from models import CAE
    cae = CAE((30, 300, 1), "")


def max_words_per_caption():
    datasets_names = ["mnist1k-3x", "oxford-102-flowers", "cub-200-2011", "flickr30k", "coco-train-2014"]
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w{3,}")
    stopwords = set(nltk.corpus.stopwords.words('english'))
    print(stopwords)

    for dataset_name in datasets_names[:]:
        dataset_path = pathlib.Path("datasets/{}".format(dataset_name))
        meta_file_path = os.path.join(dataset_path, "meta.json")
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
