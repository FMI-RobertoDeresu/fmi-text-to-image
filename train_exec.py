import itertools
import traceback
import argparse
import subprocess

parser = argparse.ArgumentParser()
# parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-model", help="model name", default="vae")
# parser.add_argument("-dataset", help="dataset name", default="mnist1k")
parser.add_argument("-dataset", help="dataset name", default="mnist30k")
parser.add_argument("-use-tpu", help="use tpu", action="store_true")
parser.add_argument("-gpus", help="number of gpus to use tpu", type=int, default=None)


def main():
    args = parser.parse_args()

    optimizer_indexes = range(10)
    loss_indexes = range(1)
    batch_size_indexes = range(1)

    for optimizer, loss, batch_size in itertools.product(optimizer_indexes, loss_indexes, batch_size_indexes):
        try:
            subproc_args = [
                "python", "train.py",
                "-model", args.model,
                "-dataset", args.dataset,
                "-optimizer-index", str(optimizer),
                "-loss-index", str(loss),
                "-batch-size-index", str(batch_size),
            ]

            if args.use_tpu:
                subproc_args.extend(["-use-tpu"])

            if args.gpus is not None:
                subproc_args.extend(["-gpus", str(args.gpus)])

            retcode = subprocess.call(subproc_args)
            print("Code={}".format(retcode))
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
