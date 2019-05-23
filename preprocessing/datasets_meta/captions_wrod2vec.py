import os
import json
import utils
import pathlib
import sys
from multiprocessing import Pool


def save_word2vec_captions(dataset_name):
    dataset_path = pathlib.Path("../../datasets/{}".format(dataset_name))
    meta_file_path = os.path.join(dataset_path, "meta.json")
    word2vec_captions_file_path = os.path.join(dataset_path, "word2vec-captions.json")
    meta = utils.json_utils.load(meta_file_path)

    meta_captions = []
    for index, meta_entry in enumerate(meta):
        for caption in meta_entry["captions"]:
            meta_captions.append((index, caption))

    indexes, captions = list(zip(*meta_captions))

    captions_size_mb = sys.getsizeof(captions) / (1024 * 1024)
    print("Captions size for dataset {} is {} MB".format(dataset_name, captions_size_mb))

    captions_word2vec = utils.word2vec_utils.word2vec(captions)

    word2vec_captions = [[] for _ in range(len(meta))]
    for index, caption_word2vec in zip(indexes, captions_word2vec):
        word2vec_captions[index].append(caption_word2vec)

    pathlib.Path(word2vec_captions_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(word2vec_captions_file_path, 'w+') as f:
        json.dump(word2vec_captions, f)


if __name__ == "__main__":
    datasets_names = ["oxford-102-flowers", "cub-200-2011", "flickr30k"]
    with Pool(processes=1) as pool:
        pool.map(save_word2vec_captions, datasets_names, chunksize=1)
    # for dataset_name in datasets_names:
    #     save_word2vec_captions(dataset_name)
