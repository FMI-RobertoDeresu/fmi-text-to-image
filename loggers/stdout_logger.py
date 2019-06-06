import sys
import utils
import os
import pathlib


class StdoutLogger:
    def __init__(self, log_dir):
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, "{}.txt".format(utils.uid())), "a")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)
