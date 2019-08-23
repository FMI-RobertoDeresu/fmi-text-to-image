from tf_imports import tf, K, Callback


class OutputCheckpoint(Callback):
    def __init__(self, tensor_board_writer, val_data, test_data_input_word2vec, print_every=1):
        super().__init__()
        self.writer = tensor_board_writer
        self.val_data = val_data
        self.test_data_input_word2vec = test_data_input_word2vec
        self.print_every = print_every

        self.epoch = 0
        self.train_end = False

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if epoch % self.print_every == 0:
            self.log_test_results()

    def on_train_end(self, logs=None):
        self.train_end = True
        self.log_test_results()
        self.log_validation_results()

    def log_validation_results(self):
        x, y = self.val_data
        summary = K.eval(tf.summary.image(name="eval_real", tensor=y[:4], max_outputs=4))
        self.writer.add_summary(summary, global_step=self.epoch)

        outputs = self.model.predict(x=x[:4], batch_size=128)
        summary = K.eval(tf.summary.image(name="eval_gen", tensor=outputs, max_outputs=4))
        self.writer.add_summary(summary, global_step=self.epoch)

        self.writer.flush()

    def log_test_results(self):
        outputs = self.model.predict(x=self.test_data_input_word2vec, batch_size=128)

        if self.train_end:
            name = "test_end_training".format(self.epoch + 1)
        else:
            name = "test_epoch_{}".format(self.epoch + 1)

        summary = K.eval(tf.summary.image(name=name, tensor=outputs, max_outputs=4))
        self.writer.add_summary(summary, global_step=self.epoch)
        self.writer.flush()
