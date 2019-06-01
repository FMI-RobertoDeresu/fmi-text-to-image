import pathlib
import os
import utils
import nltk
from models import CAE


def main():
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


if __name__ == '__main__':
    main()
    # max_words_per_caption()
