from tf_imports import TensorBoard


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
