import json


def load(file_path):
    with open(str(file_path)) as f:
        data = json.load(f)
    return data


def dump(data, file_path, indent=4):
    with open(str(file_path), 'w+') as f:
        json.dump(data, f, indent=indent)


def dumps(data, indent=4):
    return json.dumps(data, indent=indent)
