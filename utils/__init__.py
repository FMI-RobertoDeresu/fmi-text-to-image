import time
from utils import json_utils, pickle_utils, plot_utils


def uid():
    return str(time.time()).replace(".", "").ljust(17, "0")
