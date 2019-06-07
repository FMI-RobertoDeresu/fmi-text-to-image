import utils
from PIL import Image
from pathlib import Path

resize = True
raw_dataset_path = Path("../../tmp/datasets/birds")
processed_dataset_path = Path("../../datasets/cub-200-2011-128x128")

images_dirs_path = Path(raw_dataset_path, "images")
captions_dirs_path = Path(raw_dataset_path, "captions")
meta = []

for dir_path in list(images_dirs_path.glob(pattern="*")):
    dir_name = dir_path.name
    images_paths = list(dir_path.glob(pattern="*"))
    for image_path in images_paths:
        image_name = image_path.name
        captions_file_name = image_path.name[:-4] + ".txt"
        captions_file_path = Path(captions_dirs_path, dir_name, captions_file_name)
        with open(captions_file_path, 'r') as f:
            image_captions = [x.rstrip("\n\r") for x in f.readlines()]

        label = int(dir_name[:3])

        img = Image.open(str(image_path))
        if resize:
            img = img.resize((128, 128), Image.ANTIALIAS)

        new_file_path = Path(processed_dataset_path, "images", dir_name, image_name)
        new_file_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(new_file_path))

        meta.append({
            "image": Path("images", dir_name, image_name).as_posix(),
            "label": label,
            "captions": image_captions
        })

meta_file_path = Path(processed_dataset_path, "meta.json")
meta_file_path.parent.mkdir(parents=True, exist_ok=True)
utils.json_utils.dump(meta, meta_file_path)
