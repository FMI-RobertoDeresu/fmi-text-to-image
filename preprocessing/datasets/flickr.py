import os
import json
import pathlib
import csv
import itertools
from PIL import Image

resize = False
raw_dataset_path = pathlib.Path("../../tmp/datasets/flickr30k")
processed_dataset_path = pathlib.Path("../../datasets/flickr30k-original-size")

meta = []

captions_path = os.path.join(raw_dataset_path, "results.csv")
with open(captions_path, encoding="utf8") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter='|')
    rows = list(csv_reader)
    rows_grouped_by_image = itertools.groupby(rows, key=lambda x: x["image_name"])
    for image_name, image_rows in rows_grouped_by_image:
        image_captions = list(map(lambda x: x["comment"], image_rows))

        old_file_path = os.path.join(raw_dataset_path, "images", image_name)
        new_file_path = os.path.join(processed_dataset_path, "images", image_name)

        img = Image.open(old_file_path)
        if resize:
            img = img.resize((128, 128), Image.ANTIALIAS)

        pathlib.Path(new_file_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(new_file_path)

        meta.append({
            "image": os.path.join("images", image_name),
            "label": None,
            "captions": image_captions
        })

meta_file_path = os.path.join(processed_dataset_path, "meta.json")
pathlib.Path(meta_file_path).parent.mkdir(parents=True, exist_ok=True)
with open(meta_file_path, 'w+') as f:
    json.dump(meta, f, indent=4)
