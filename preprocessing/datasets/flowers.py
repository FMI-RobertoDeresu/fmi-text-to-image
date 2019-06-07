import utils
from PIL import Image
from pathlib import Path

resize = False
flowers_raw_dataset_path = Path("../../tmp/datasets/flowers")
flowers_processed_dataset_path = Path("../../datasets/flowers")

labels_file_path = Path(flowers_raw_dataset_path, "labels.txt")
with open(labels_file_path, 'r') as f:
    labels = [int(x) for x in f.readlines()]

captions = dict()
captions_files_paths = list(Path(flowers_raw_dataset_path, "text").glob(pattern="**/*.txt"))
for caption_file_path in captions_files_paths:
    image_name = caption_file_path.name.split(".")[0]
    with open(str(caption_file_path), 'r') as f:
        captions[image_name] = [x.rstrip("\n\r") for x in f.readlines()]

meta = []

images_dir_path = Path(flowers_raw_dataset_path, "jpg")
image_files_paths = list(images_dir_path.glob(pattern="*"))

for index, image_file_path in enumerate(image_files_paths):
    image_name = image_file_path.name.split(".")[0]
    label_folder_name = str(labels[index]).rjust(4, "0")

    img = Image.open(str(image_file_path))
    if resize:
        img = img.resize((128, 128), Image.ANTIALIAS)

    new_file_path = Path(flowers_processed_dataset_path, "images", label_folder_name, image_file_path.name)
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(new_file_path))

    meta.append({
        "image": Path("images", label_folder_name, image_file_path.name).as_posix(),
        "label": labels[index],
        "captions": captions[image_name]
    })

meta_file_path = Path(flowers_processed_dataset_path, "meta.json")
meta_file_path.parent.mkdir(parents=True, exist_ok=True)
utils.json_utils.dump(meta, meta_file_path)
