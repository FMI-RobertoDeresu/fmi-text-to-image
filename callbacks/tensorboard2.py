import utils
from tf_imports import tf, K, TensorBoard


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

    def on_train_begin(self, logs=None):
        optimizer_config = { "name":self.model.optimizer.__class__.__name__}
        optimizer_config.update(self.model.optimizer.get_config())
        optimizer_config_json = utils.json_utils.dumps(optimizer_config)
        optimizer_config_tensor = tf.convert_to_tensor(optimizer_config_json)

        summary = K.eval(tf.summary.text(name="optimizer vars", tensor=optimizer_config_tensor))
        self.writer.add_summary(summary, global_step=0)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
