import os
import argparse
import const
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model name", default="cae")
parser.add_argument("dataset", help="dataset name", default="mnist")

if __name__ == "__main__":
    args = parser.parse_args()
    model = models.models_dict[args.model]
    dataset_meta = utils.json_utils.load(os.path.join(const.datasets[args.dataset], "meta.json"))

    model.train(dataset_meta)
