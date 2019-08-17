import math
from tf_imports import Callback


class CheckNanLoss(Callback):
    def __init__(self, monitor='val_loss'):
        super(CheckNanLoss, self).__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if math.isnan(current):
            self.model.stop_training = True