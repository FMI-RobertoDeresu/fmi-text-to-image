import utils
from pathlib import Path
from tf_imports import K, tf_summary
from tf_imports import EarlyStopping
from abc import ABC, abstractmethod
from callbacks import OutputCheckpoint, TensorBoard2, CheckNanLoss
from keras.utils import plot_model


class BaseModel(ABC):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.optimizer = None
        self.loss = None

        model = self._create_model(input_shape)
        # model.summary()

        self.model = model
        self.model_compiled = False

    @abstractmethod
    def _create_model(self, input_shape):
        pass

    @abstractmethod
    def plot_model(self, save_to_dir):
        pass

    @staticmethod
    def _plot_model(model, file_path):
        plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
        print(model.summary())

    def compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self._compile(optimizer, loss)

    @abstractmethod
    def _compile(self, optimizer, loss):
        pass

    def train(self, x, y, batch_size, out_folder, output_checkpoint_inputs_word2vec=None):
        if not self.model_compiled:
            raise Exception("The model must be compiled first.")

        train_uid = utils.uid()
        description = "{train_uid} batch={batch_size}".format(train_uid=train_uid, batch_size=batch_size)
        print("Training: {}".format(description))

        # callbacks
        callbacks = []

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.02,
            patience=20,
            verbose=1,
            mode='min',
            restore_best_weights=True)
        callbacks.append(early_stopping)

        check_nan = CheckNanLoss('val_loss')
        callbacks.append(check_nan)

        tensor_board_log_dir = Path(out_folder, "tensorboard", description)
        tensor_board_log_dir.mkdir(parents=True, exist_ok=True)
        tensor_board_writer = tf_summary.FileWriter(str(tensor_board_log_dir), K.get_session().graph)

        if output_checkpoint_inputs_word2vec is not None:
            output_checkpoint = OutputCheckpoint(
                tensor_board_writer=tensor_board_writer,
                val_data=(x[:4], y[:4]),
                test_data_input_word2vec=output_checkpoint_inputs_word2vec,
                print_every=30)
            callbacks.append(output_checkpoint)

        # last because close the writer on training end
        tensor_board = TensorBoard2(writer=tensor_board_writer)
        callbacks.append(tensor_board)

        # fit
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=300,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_split=0.2)

        # save
        weights_path = Path(out_folder, "weights", "{}.h5".format(description))
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        # self.model.save_weights(str(weights_path))

    def load_weights(self, weights_file_path):
        if not Path(weights_file_path).exists():
            raise Exception("weights file '{}' not found.".format(weights_file_path))

        self.model.load_weights(str(weights_file_path))

    def predict(self, x_predict):
        prediction = self.model.predict(x=x_predict, batch_size=128)
        return prediction
