import os
import json
import pathlib
from PIL import Image

resize = False
flowers_raw_dataset_path = pathlib.Path("../../tmp/datasets/flowers")
flowers_processed_dataset_path = pathlib.Path("../../datasets/flowers")

labels_file_path = os.path.join(flowers_raw_dataset_path, "labels.txt")
with open(labels_file_path, 'r') as f:
    labels = [int(x) for x in f.readlines()]

captions = dict()
text_class_dirs = list(os.walk(os.path.join(flowers_raw_dataset_path, "text")))[1:]
for root, _, file_names in text_class_dirs:
    for file_name in file_names:
        file_path = os.path.join(root, file_name)
        image_name = file_name.split(".")[0]
        with open(file_path, 'r') as f:
            captions[image_name] = [x.rstrip("\n\r") for x in f.readlines()]

meta = []

images_dirs_path = os.path.join(flowers_raw_dataset_path, "jpg")
image_files = list(os.walk(images_dirs_path))[0][2]
for index, image_file_name in enumerate(image_files):
    image_name = image_file_name.split(".")[0]
    label_folder_name = str(labels[index]).rjust(4, "0")

    old_file_path = os.path.join(images_dirs_path, image_file_name)
    new_file_path = os.path.join(flowers_processed_dataset_path, label_folder_name, image_file_name)
    img = Image.open(old_file_path)
    if resize:
        img = img.resize((128, 128), Image.ANTIALIAS)

    pathlib.Path(new_file_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(new_file_path)

    meta.append({
        "image": os.path.join(label_folder_name, image_file_name),
        "label": labels[index],
        "captions": captions[image_name]
    })

meta_file_path = os.path.join(flowers_processed_dataset_path, "meta.json")
pathlib.Path(meta_file_path).parent.mkdir(parents=True, exist_ok=True)
with open(meta_file_path, 'w+') as f:
    json.dump(meta, f, indent=4)
