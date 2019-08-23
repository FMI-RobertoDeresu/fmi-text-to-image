import itertools
import traceback
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
# parser.add_argument("-model", help="model name", default="vae")
parser.add_argument("-dataset", help="dataset name", default="mnist1k")
# parser.add_argument("-dataset", help="dataset name", default="mnist30k")
parser.add_argument("-trainings", help="number of trainings", default="3")


def main():
    args = parser.parse_args()

    optimizer_indexes = range(1)
    loss_indexes = range(1)
    batch_size_indexes = range(1)
    lr_schedule_fn_indexes = range(1)

    for optimizer, loss, batch_size, lr_schedule_fn in \
            itertools.product(optimizer_indexes, loss_indexes, batch_size_indexes, lr_schedule_fn_indexes):
        for _ in range(int(args.trainings)):
            try:
                subproc_args = [
                    "python", "train.py",
                    "-model", args.model,
                    "-dataset", args.dataset,
                    "-optimizer-index", str(optimizer),
                    "-loss-index", str(loss),
                    "-batch-size-index", str(batch_size),
                    "-lr-schedule-fn-index", str(lr_schedule_fn)
                ]

                retcode = subprocess.call(subproc_args)
                print("Code={}".format(retcode))
            except Exception:
                traceback.print_exc()


if __name__ == '__main__':
    main()
