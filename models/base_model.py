import utils
from pathlib import Path
from tf_imports import TensorBoard, EarlyStopping
from abc import ABC, abstractmethod
from tf_imports import tf, K, multi_gpu_model, tf_summary
from callbacks import OutputCheckpoint, TensorBoard2


class BaseModel(ABC):
    def __init__(self, input_shape, use_tpu=False, gpus=None):
        if use_tpu and gpus > 0:
            raise Exception("If tpus are used, gpus must be 0.")

        if gpus is not None and gpus < 2:
            raise Exception("If gpus is specified, then it must be and integer >= 2.")

        self.input_shape = input_shape
        self.optimizer = None
        self.loss = None

        model = self._create_model(input_shape)
        # model.summary()

        # model as tpu
        if use_tpu:
            model = tf.contrib.tpu.keras_to_tpu_model(
                model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu='demo-tpu')
                )
            )

        # model as gpu
        if (gpus or 0) > 1:
            model = multi_gpu_model(model, gpus=gpus, cpu_relocation=True)

        self.model = model
        self.model_compiled = False

    @abstractmethod
    def _create_model(self, input_shape):
        pass

    def compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self._compile(optimizer, loss)

    @abstractmethod
    def _compile(self, optimizer, loss):
        pass

    def train(self, x, y, batch_size, out_folder, output_checkpoint_inputs):
        if not self.model_compiled:
            raise Exception("The model must be compiled first.")

        train_uid = utils.uid()
        description = "{train_uid} opt={optimizer} loss={loss} batch={batch_size}".format(
            train_uid=train_uid,
            optimizer=self.optimizer.__class__.__name__,
            loss=self.loss.__name__,
            batch_size=batch_size)
        print("Training: {}".format(description))

        # callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=1,
            mode='min',
            restore_best_weights=True)

        tensor_board_log_dir = Path(out_folder, "tensorboard", description)
        tensor_board_log_dir.mkdir(parents=True, exist_ok=True)
        tensor_board_writer = tf_summary.FileWriter(str(tensor_board_log_dir), K.get_session().graph)
        tensor_board = TensorBoard2(writer=tensor_board_writer)

        output_checkpoint = OutputCheckpoint(
            tensor_board_writer=tensor_board_writer,
            inputs=output_checkpoint_inputs,
            print_every=3)

        callbacks = [early_stopping, tensor_board, output_checkpoint]

        # fit
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=500,
            # epochs=7,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_split=0.2)

        # save
        weights_path = Path(out_folder, "weights", "{}.h5".format(description))
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(weights_path))

    def load_weights(self, weights_file_path):
        if not Path(weights_file_path).exists():
            raise Exception("weights file '{}' not found.".format(weights_file_path))

        self.model.load_weights(str(weights_file_path))

    def predict(self, x_predict):
        prediction = self.model.predict(x=x_predict, batch_size=128)
        return prediction
