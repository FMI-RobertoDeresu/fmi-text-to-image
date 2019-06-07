import itertools
import traceback
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model name", default="cae")
parser.add_argument("-dataset", help="dataset name", default="mnist10k")
parser.add_argument("-use-tpu", help="use tpu", action="store_true")
parser.add_argument("-gpus", help="number of gpus to use tpu", type=int, default=None)


def main():
    args = parser.parse_args()

    optimizer_indexes = range(4)
    loss_indexes = range(2)
    batch_size_indexes = range(5)

    for optimizer, loss, batch_size, use_dense_layers in itertools.product(optimizer_indexes, loss_indexes,
                                                                           batch_size_indexes, [True, False]):
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

            if use_dense_layers:
                subproc_args.extend(["-use-dense-layers"])

            retcode = subprocess.call(subproc_args)
            print("Code={}".format(retcode))
        except Exception:
            traceback.print_exc()

    # subproc_args = [
    #     "python", "predict.py",
    #     "-model", args.model,
    #     "-dataset", args.dataset,
    #     "-weights", "all",
    #     "-word2vec", "local",
    # ]
    #
    # retcode = subprocess.call(subproc_args)
    # print("Code={}".format(retcode))


if __name__ == '__main__':
    main()
