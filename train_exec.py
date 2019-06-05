import itertools
import traceback
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist10k")

parser.add_argument("-use-tpu", help="use tpu", action="store_true")


def main():
    args = parser.parse_args()

    optimizer_indexes = range(5)
    loss_indexes = range(9)
    batch_size_indexes = range(3)

    for optimizer, loss, batch_size in itertools.product(optimizer_indexes, loss_indexes, batch_size_indexes):
        try:
            subproc_args = [
                "python", "train.py",
                "-model", args.model,
                "-dataset", args.dataset,
                "-optimizer-index", str(optimizer),
                "-loss-index", str(loss),
                "-batch-size-index", str(batch_size)
            ]

            if args.use_tpu:
                subproc_args.extend(["-use-tpu"])

            retcode = subprocess.call(subproc_args)
            print("Code={}".format(retcode))
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
