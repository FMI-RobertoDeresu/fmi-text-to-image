import json


def load(file_name):
    with open(file_name) as f:
        json_data = json.load(f)
    return json_data
