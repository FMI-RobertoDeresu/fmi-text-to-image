import utils
import csv
import itertools
from PIL import Image
from pathlib import Path

resize = False
raw_dataset_path = Path("../../tmp/datasets/flickr30k")
processed_dataset_path = Path("../../datasets/flickr30k-original-size")

meta = []

captions_path = Path(raw_dataset_path, "results.csv")
with open(captions_path, encoding="utf8") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter='|')
    rows = list(csv_reader)
    rows_grouped_by_image = itertools.groupby(rows, key=lambda x: x["image_name"])
    for image_name, image_rows in rows_grouped_by_image:
        image_captions = list(map(lambda x: x["comment"], image_rows))

        old_file_path = Path(raw_dataset_path, "images", image_name)

        img = Image.open(str(old_file_path))
        if resize:
            img = img.resize((128, 128), Image.ANTIALIAS)

        new_file_path = Path(processed_dataset_path, "images", image_name)
        new_file_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(new_file_path))

        meta.append({
            "image": Path("images", image_name).as_posix(),
            "label": None,
            "captions": image_captions
        })

meta_file_path = Path(processed_dataset_path, "meta.json")
meta_file_path.parent.mkdir(parents=True, exist_ok=True)
utils.json_utils.dump(meta, meta_file_path)
