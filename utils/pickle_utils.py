import pickle


def load(file_name):
    with open(file_name, "rb") as f:
        json_data = pickle.load(f)
    return json_data
