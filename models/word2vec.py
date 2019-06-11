import os
import requests
import numpy
import time
import gensim
import nltk


class Word2Vec:
    __instance = None

    @staticmethod
    def get_instance():
        if Word2Vec.__instance is None:
            Word2Vec()
        return Word2Vec.__instance

    def __init__(self):
        if Word2Vec.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Word2Vec.__instance = self

        self.model = None
        self.useNorm = False
        self.model_path = '{}/../tmp/word2vec_300.bin'.format(os.path.dirname(os.path.realpath(__file__)))
        # self.url = "http://www.robertoderesu.com/ml/word2vec/word-vec"
        self.url = "http://127.0.0.1:8001/word-vec"
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"\w{3,}")

    def get_embeddings(self, texts, print_every=None):
        if self.model is None:
            self.load_model()

        output = []
        n_texts = len(texts)
        for index, text in enumerate(texts):
            words = self.tokenizer.tokenize(text)
            words_embeddings = []

            for word in words:
                if word in self.model.vocab:
                    word_embedding = self.model.word_vec(word, self.useNorm)
                    words_embeddings.append([x.item() for x in word_embedding])

            output.append(words_embeddings)

            if print_every is not None and index % print_every == 0:
                print("Processed {:0.2f}% {}/{}".format(index / n_texts * 100, index, n_texts))

        return output

    def get_embeddings_remote(self, texts):
        n_chunks = max(int(len(texts) / 100 + 1), 1)
        # n_chunks = 1
        chunks = [list(x) for x in numpy.array_split(numpy.array(texts), n_chunks)]

        result = []
        for text_chunk in chunks:
            request_data = {
                "texts": text_chunk,
                "useNorm": False
            }
            response = requests.post(self.url, json=request_data)
            response_data = response.json()
            result.extend(response_data)

        return result

    def load_model(self):
        print("Loading model...")
        start_time_model = time.time()

        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)

        end_time_model = time.time()
        print("Model loaded! Elapsed: " + str(end_time_model - start_time_model))
