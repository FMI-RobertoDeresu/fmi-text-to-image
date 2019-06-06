import pathlib

DATASETS_PATH = {
    "mnist1k": str(pathlib.Path("./datasets/mnist1k-3x")),
    "mnist10k": str(pathlib.Path("./datasets/mnist10k-3x")),
    "mnist30k": str(pathlib.Path("./datasets/mnist30k-3x")),
    "flowers": str(pathlib.Path("./datasets/oxford-102-flowers")),
    "birds": str(pathlib.Path("./datasets/cub-200-2011")),
    "flickr": str(pathlib.Path("./datasets/flickr30k")),
    "coco": str(pathlib.Path("./datasets/coco-train-2014")),
}

INPUT_SHAPE = (3, 300, 1)