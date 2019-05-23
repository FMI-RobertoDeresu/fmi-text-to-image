import requests
import numpy

url = "http://www.robertoderesu.com/ml/word2vec/word-vec"


def word2vec(texts):
    n_chunks = max(int(len(texts) / 1000 + 1), 1)
    # n_chunks = 1
    chunks = [list(x) for x in numpy.array_split(numpy.array(texts), n_chunks)]

    result = []
    for text_chunk in chunks:
        request_data = {
            "texts": text_chunk,
            "useNorm": False
        }
        response = requests.post(url, json=request_data)
        response_data = response.json()
        result.extend(response_data)

    return result
