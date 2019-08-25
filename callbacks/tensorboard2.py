import utils
from tf_imports import tf, K, TensorBoard


class TensorBoard2(TensorBoard):
    def __init__(self, writer, histogram_freq=0, batch_size=None):
        super().__init__(
            log_dir=None,
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=False,
            write_grads=False,
            write_images=False)
        self.writer = writer
        self.log_dir = ""

    def _init_writer(self):
        pass

    def set_model(self, model):
        writer = self.writer
        super().set_model(model)
        self.writer = writer

    def on_train_begin(self, logs=None):
        self._log_optimizer_description(logs)
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = self._rename_logs(logs)
        super().on_epoch_end(epoch, logs)

    def _log_optimizer_description(self, logs):
        optimizer_config = {"name": self.model.optimizer.__class__.__name__}
        optimizer_config.update(self.model.optimizer.get_config())
        optimizer_config_json = utils.json_utils.dumps(optimizer_config, sort_keys=True)
        optimizer_config_tensor = tf.convert_to_tensor(optimizer_config_json)

        summary = K.eval(tf.summary.text(name="optimizer vars", tensor=optimizer_config_tensor))
        self.writer.add_summary(summary, global_step=0)
        self.writer.flush()

    def _rename_logs(self, logs):
        for key in logs:
            logs["performance/" + key] = logs[key]
            del logs[key]
        return logs
