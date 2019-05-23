import utils
import os
import pathlib
import re

datasets_dir_path = pathlib.Path("../../datasets")
datasets_names = list(os.walk(datasets_dir_path))[0][1]
datasets_paths = [(x, os.path.join(datasets_dir_path, x)) for x in datasets_names]

for dataset_name, dataset_path in datasets_paths:
    meta_file_name = os.path.join(dataset_path, "meta.json")
    meta = utils.json_utils.load(meta_file_name)

    max_length = 0
    for meta_entry in meta:
        for caption in meta_entry["captions"]:
            try:
                words = re.findall(r"\w+", caption)
                max_length = max(max_length, len(words))
            except:
                print(meta_entry["image"])

    print("{} - {}".format(dataset_name, max_length))
