import os
import json
import pathlib
from PIL import Image

resize = True
raw_dataset_path = pathlib.Path("../tmp/datasets/birds")
processed_dataset_path = pathlib.Path("../datasets/cub-200-2011-128x128")

images_dirs_path = os.path.join(raw_dataset_path, "images")
captions_dirs_path = os.path.join(raw_dataset_path, "captions")
meta = []

for dir_path, _, images_names in list(os.walk(images_dirs_path))[1:]:
    dir_name = os.path.basename(dir_path)
    for image_name in images_names:
        captions_file_name = image_name[:-4] + ".txt"
        captions_file_path = os.path.join(captions_dirs_path, dir_name, captions_file_name)
        with open(captions_file_path, 'r') as f:
            image_captions = [x.rstrip("\n\r") for x in f.readlines()]

        label = int(dir_name[:3])

        old_file_path = os.path.join(dir_path, image_name)
        new_file_path = os.path.join(processed_dataset_path, dir_name, image_name)

        img = Image.open(old_file_path)
        if resize:
            img = img.resize((128, 128), Image.ANTIALIAS)

        pathlib.Path(new_file_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(new_file_path)

        meta.append({
            "image": os.path.join(dir_name, image_name),
            "label": label,
            "captions": image_captions
        })

meta_file_path = os.path.join(processed_dataset_path, "meta.json")
pathlib.Path(meta_file_path).parent.mkdir(parents=True, exist_ok=True)
with open(meta_file_path, 'w+') as f:
    json.dump(meta, f, indent=4)
