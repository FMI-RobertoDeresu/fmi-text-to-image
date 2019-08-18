from tf_imports import tf, K, Callback


class OutputCheckpoint(Callback):
    def __init__(self, tensor_board_writer, inputs_word2vec, print_every=1):
        super().__init__()
        self.writer = tensor_board_writer
        self.inputs_word2vec = inputs_word2vec
        self.print_every = print_every

        self.epoch = 0
        self.train_end = False

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if epoch % self.print_every == 0:
            self.log()

    def on_train_end(self, logs=None):
        self.train_end = True
        self.log()

    def log(self):
        outputs = self.model.predict(x=self.inputs_word2vec, batch_size=128)

        if self.train_end:
            name = "output end training".format(self.epoch + 1)
        else:
            name = "output checkpoint epoch {}".format(self.epoch + 1)

        summary = K.eval(tf.summary.image(name=name, tensor=outputs, max_outputs=4))
        self.writer.add_summary(summary, global_step=self.epoch)
        self.writer.flush()
