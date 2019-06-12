from tf_imports import K, TensorBoard


class TensorBoard2(TensorBoard):
    def __init__(self, writer):
        super().__init__(
            log_dir=None,
            write_graph=True,
            write_grads=True,
            write_images=True)
        self.writer = writer

    def _init_writer(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'beta_1': K.eval(self.model.optimizer.beta_1)})
        logs.update({'beta_2': K.eval(self.model.optimizer.beta_2)})
        logs.update({'decay': K.eval(self.model.optimizer.decay)})
        super().on_epoch_end(epoch, logs)
