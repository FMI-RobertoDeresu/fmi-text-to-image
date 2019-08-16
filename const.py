from pathlib import Path

DATASETS_PATH = {
    "mnist1k": str(Path("./datasets/mnist1k-3x")),
    "mnist10k": str(Path("./datasets/mnist10k-3x")),
    "mnist30k": str(Path("./datasets/mnist30k-3x")),
    "flowers": str(Path("./datasets/oxford-102-flowers")),
    "birds": str(Path("./datasets/cub-200-2011")),
    "flickr": str(Path("./datasets/flickr30k")),
    "coco": str(Path("./datasets/coco-train-2014")),
}

INPUT_SHAPE = (30, 300, 1)
OUTPUT_IMAGE_SIZE = (128, 128)
NOISE_STDDEV = 0.01

OUTPUT_CHECKPOINT_INPUTS = {
    "mnist1k": [
        "zero one two",
        "three four five",
        "six seven eight"
    ],
    "mnist30k": [
        "zero one two",
        "three four five",
        "six seven eight"
    ],
    "flowers": [
        "flower is blue and green in color",
        "this flower is white and yellow",
        "this flower has a long pointed red petal"
    ]
}
