import os
import pathlib
import utils
import numpy as np
import time
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import EarlyStopping


class CAE:
    def __init__(self, input_shape, dataset_name, use_tpu):
        self.input_shape = input_shape
        self.dataset_name = dataset_name
        self.print_model_summary = False

        self.train_results_path = str(pathlib.Path("tmp/train/cae/{}/results.json".format(dataset_name)))
        self.weights_path = str(pathlib.Path("tmp/train/cae/{}/weights/{{out_folder}}.hdf5".format(dataset_name)))
        self.tensor_board_log_dir = str(pathlib.Path('tmp/tensorboard/cae/{}/{{out_folder}}'.format(dataset_name)))

        self._create_model(input_shape, use_tpu)

    def _create_model(self, input_shape, use_tpu):
        # N, M, _ = input_shape
        # input
        input_layer = Input(shape=input_shape)  # (N, M, 1)

        # noise
        # input_layer = GaussianNoise(0.1)(input_layer)  # (N, M, 1)

        # encoder
        encoder = Conv2D(2, (3, 3), activation='relu', padding='same')(input_layer)  # (N, M, 2)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/2 , M/2, 2)

        encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)  # (N/2 , M/2, 4)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/4 , M/4, 4)

        encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)  # (N/4 , M/4, 8)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/8 , M/8, 8)

        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)  # (N/8 , M/8, 16)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/16 , M/16, 16)

        encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)  # (N/16 , M/16, 32)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/32 , M/32, 32)

        encoder = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)  # (N/32 , M/32, 64)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/64 , M/64, 64)

        encoder = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)  # (N/64 , M/64, 128)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/128 , M/128, 128)

        encoder = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder)  # (N/128 , M/128, 256)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/256 , M/256, 256)

        encoder = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)  # (N/256 , M/256, 512)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/512 , M/512, 512)
        # Shape here must be: (1, 1, 512)

        # decoder
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)  # (1 , 1, 512)
        decoder = UpSampling2D((2, 2))(decoder)  # (2, 2, 512)

        decoder = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(decoder)  # (2, 2, 256)
        decoder = UpSampling2D((2, 2))(decoder)  # (4, 4, 256)

        decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)  # (4, 4, 128)
        decoder = UpSampling2D((2, 2))(decoder)  # (8, 8, 128)

        decoder = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)  # (8, 8, 64)
        decoder = UpSampling2D((2, 2))(decoder)  # (16, 16, 64)

        decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)  # (16, 16, 32)
        decoder = UpSampling2D((2, 2))(decoder)  # (32, 32, 32)

        decoder = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder)  # (32, 32, 316)
        decoder = UpSampling2D((2, 2))(decoder)  # (64, 64, 16)

        decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)  # (64, 64, 8)
        decoder = UpSampling2D((2, 2))(decoder)  # (128, 128, 8)

        decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)  # (128, 128,  3)

        self.model = Model(input_layer, decoder)
        if self.print_model_summary:
            self.model.summary()

        if use_tpu is not None:
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver()
                )
            )

    def train(self, x_train, y_train, x_test, y_test, optimizer, loss, batch_size):
        train_uid = utils.uid()
        description = "{train_uid} opt={optimizer} loss={loss} batch={batch_size}".format(
            train_uid=train_uid,
            optimizer=optimizer.__class__.__name__,
            loss=loss.__name__,
            batch_size=batch_size)

        print("Training: {}".format(description))
        weights_path = pathlib.Path(self.weights_path.format(out_folder=description))
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        tensor_board_log_dir = pathlib.Path(self.tensor_board_log_dir.format(out_folder=description))
        tensor_board_log_dir.mkdir(parents=True, exist_ok=True)

        # callbacks
        checkpoint_weights = ModelCheckpoint(
            filepath=str(weights_path),
            verbose=0,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min')

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=3,
            verbose=1,
            mode='min',
            restore_best_weights=True)

        tensor_board = TensorBoard(log_dir=str(tensor_board_log_dir))

        callbacks = [checkpoint_weights, tensor_board, early_stopping]

        # compile
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # fit
        start_time = time.time()
        fit = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=500,
            # epochs=1,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_split=0.2)

        # evaluate
        evaluate = self.model.evaluate(x=x_test, y=y_test, verbose=0)
        loss, accuracy = evaluate[:2]

        # save results
        self._save_result(train_uid, description, start_time, fit, loss, accuracy, early_stopping)

    def _save_result(self, train_uid, description, start_time, fit, loss, accuracy, early_stopping):
        if os.path.isfile(self.train_results_path):
            train_results = utils.json_utils.load(self.train_results_path)
        else:
            train_results = {
                "model": self.__class__.__name__,
                "dataset": self.dataset_name,
                "input_shape": self.input_shape,
                "training_sessions": []
            }

        train_results["training_sessions"].append({
            "uid": train_uid,
            "description": description,
            "start_time": str(datetime.fromtimestamp(start_time)),
            "end_time": str(datetime.now()),
            "elapsed_time": str(datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)),
            "trained_epochs": len(fit.epoch),
            "model": {
                "optimizer": fit.model.optimizer.__class__.__name__,
                "loss": fit.model.loss_functions[0].__name__
            },
            "evaluation": {
                "loss": loss,
                "acc": accuracy,
            },
            "fit_params": {
                "batch_size": fit.params["batch_size"],
                "epochs": fit.params["epochs"]
            },
            "best_checkpoint_path": self.weights_path,
            "early_stopping": {
                "monitor": early_stopping.monitor,
                "min_delta": early_stopping.min_delta,
                "patience": early_stopping.patience
            }
        })

        utils.json_utils.dump(train_results, self.train_results_path)

    def load_weights(self, mode="last"):
        train_results = utils.json_utils.load(self.train_results_path)
        train_session = train_results["training_sessions"]

        if mode == "last":
            weights_file_path = train_session[-1]["best_checkpoint_path"]
        elif mode == "min_loss":
            results_losses = [result["evaluation"]["loss"] for result in train_session]
            min_loss_index = np.argmin(results_losses)
            weights_file_path = train_session[min_loss_index]["best_checkpoint_path"]
        elif mode == "max_acc":
            results_accs = [result["evaluation"]["acc"] for result in train_session]
            max_acc_index = np.argmax(results_accs)
            weights_file_path = train_session[max_acc_index]["best_checkpoint_path"]
        else:
            raise Exception("'{}' mode not implemented.".format(mode))

        if not os.path.isfile(weights_file_path):
            raise Exception("weights file '{}' not found.".format(weights_file_path))

        self.model.load_weights(weights_file_path)

    def predict(self, x_predict):
        prediction = self.model.predict(x=x_predict, batch_size=128)
        return prediction
