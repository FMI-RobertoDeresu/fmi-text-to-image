import pathlib

DATASETS_PATH = {
    "mnist1k": pathlib.Path("./datasets/mnist1k-3x"),
    "mnist10k": pathlib.Path("./datasets/mnist10k-3x"),
    "mnist30k": pathlib.Path("./datasets/mnist30k-3x"),
    "flowers": pathlib.Path("./datasets/oxford-102-flowers"),
    "birds": pathlib.Path("./datasets/cub-200-2011"),
    "flickr": pathlib.Path("./datasets/flickr30k"),
    "coco": pathlib.Path("./datasets/coco-train-2014"),
}

INPUT_SHAPE = (3, 300, 1)