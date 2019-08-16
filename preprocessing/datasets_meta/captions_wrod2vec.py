import utils
import pickle
import time
from pathlib import Path
from models.word2vec import Word2Vec


def save_word2vec_captions(word2vec, dataset_name):
    dataset_path = Path("../../datasets/{}".format(dataset_name))
    meta_file_path = Path(dataset_path, "meta.json")
    meta = utils.json_utils.load(meta_file_path)

    meta_captions = []
    max_no_of_words = 0
    max_no_of_word2vec_words = 0
    for index, meta_entry in enumerate(meta):
        for caption in meta_entry["captions"]:
            captions_words = caption.split()
            captions_words = list(filter(lambda x: x not in utils.stopwords, captions_words))
            captions_words = list(filter(lambda x: len(x) > 2, captions_words))
            caption = " ".join(captions_words)
            meta_captions.append((index, caption))

            max_no_of_words = max(max_no_of_words, len(captions_words))

            no_of_word2vec_words = len(list(filter(lambda x: x in word2vec.model.vocab, captions_words)))
            max_no_of_word2vec_words = max(max_no_of_word2vec_words, no_of_word2vec_words)

    indexes, captions = list(zip(*meta_captions))

    print("Max number of words in caption: {}".format(max_no_of_words))
    print("Max number of word2vec words in caption: {}".format(max_no_of_word2vec_words))

    captions_word2vec = word2vec.get_embeddings(captions, print_every=1000)

    word2vec_captions = [[] for _ in range(len(meta))]
    for index, caption_word2vec in zip(indexes, captions_word2vec):
        word2vec_captions[index].append(caption_word2vec)

    word2vec_captions_bin_file_path = Path(dataset_path, "word2vec-captions-nostopwords.bin")
    word2vec_captions_bin_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(word2vec_captions_bin_file_path, 'wb') as f2:
        pickle.dump(word2vec_captions, f2)


def main():
    datasets_names = ([
        "mnist1k-3x",
        "mnist10k-3x",
        "mnist30k-3x",
        "oxford-102-flowers",
        "cub-200-2011",
        "flickr30k",
        "coco-train-2014"
    ])[2:3]

    word2vec = Word2Vec.get_instance()
    word2vec.load_model()

    for dataset_name in datasets_names:
        print("Processing {}".format(dataset_name))
        start_time = time.time()
        save_word2vec_captions(word2vec, dataset_name)
        print("Elapsed: {:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
