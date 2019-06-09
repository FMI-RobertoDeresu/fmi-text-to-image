import pickle


def load(file_path):
    with open(str(file_path), "rb") as f:
        json_data = pickle.load(f)
    return json_data
