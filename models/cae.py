import os
import utils
import time
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tf_imports import Model, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten, Dense, Reshape
from tf_imports import TensorBoard, ModelCheckpoint, EarlyStopping, optimizers


class CAE:
    def __init__(self, input_shape, out_folder, use_tpu, use_dense_layers):
        self.input_shape = input_shape
        self.out_folder = out_folder
        self.print_model_summary = False

        self.train_results_path = str(Path("{}/results.json".format(out_folder)))
        self.weights_path = str(Path("{}/weights/{{weights_name}}.h5".format(out_folder)))
        self.tensor_board_log_dir = str(Path('{}/tensorboard/{{folder_name}}'.format(out_folder)))

        self._create_model(input_shape, use_tpu, use_dense_layers)

    def _create_model(self, input_shape, use_tpu, use_dense_layers):
        # N, M, _ = input_shape
        # input
        input_layer = Input(shape=input_shape)  # (N, M, 1)
        input_drop_layer = Dropout(0.2)(input_layer)  # (N, M, 1)

        # encoder
        encoder = Conv2D(2, (3, 3), padding='same', activation='relu')(input_drop_layer)  # (N, M, 2)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/2 , M/2, 2)

        encoder = Conv2D(4, (3, 3), padding='same', activation='relu')(encoder)  # (N/2 , M/2, 4)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/4 , M/4, 4)

        encoder = Conv2D(8, (3, 3), padding='same', activation='relu')(encoder)  # (N/4 , M/4, 8)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/8 , M/8, 8)

        encoder = Conv2D(16, (3, 3), padding='same', activation='relu')(encoder)  # (N/8 , M/8, 16)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/16 , M/16, 16)

        encoder = Conv2D(32, (3, 3), padding='same', activation='relu')(encoder)  # (N/16 , M/16, 32)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/32 , M/32, 32)

        encoder = Conv2D(64, (3, 3), padding='same', activation='relu')(encoder)  # (N/32 , M/32, 64)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/64 , M/64, 64)

        encoder = Conv2D(128, (3, 3), padding='same', activation='relu')(encoder)  # (N/64 , M/64, 128)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/128 , M/128, 128)

        encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder)  # (N/128 , M/128, 256)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/256 , M/256, 256)

        encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)  # (N/256 , M/256, 512)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (N/512 , M/512, 512)

        encoder = Flatten()(encoder)  # (512)

        if use_dense_layers:
            encoder = Dense(512)(encoder)  # (512)
            encoder = Dropout(0.2)(encoder)  # (512)
            encoder = Dense(512)(encoder)  # (512)

        # decoder
        decoder = Reshape((1, 1, 512))(encoder)  # (1, 1, 512)
        decoder = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(decoder)  # (2, 2, 512)
        decoder = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(decoder)  # (4, 4, 256)
        decoder = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(decoder)  # (8, 8, 128)
        decoder = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(decoder)  # (16, 16, 64)
        decoder = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(decoder)  # (32, 32, 32)
        decoder = Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(decoder)  # (64, 64, 16)
        decoder = Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(decoder)  # (128, 128, 8)
        decoder = Conv2DTranspose(3, 3, strides=1, padding='same', activation='relu')(decoder)  # (128, 128,  3)

        self.model = Model(input_layer, decoder)
        if self.print_model_summary:
            self.model.summary()

        if use_tpu:
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu='demo-tpu')
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
        weights_path = str(Path(self.weights_path.format(weights_name=description)))
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)

        tensor_board_log_dir = str(Path(self.tensor_board_log_dir.format(folder_name=description)))
        Path(tensor_board_log_dir).mkdir(parents=True, exist_ok=True)

        # callbacks
        checkpoint_weights = ModelCheckpoint(
            filepath=weights_path,
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

        tensor_board = TensorBoard(log_dir=tensor_board_log_dir)

        callbacks = [early_stopping]

        # compile
        optimizer = optimizers.Adam(clipnorm=5.)
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

        self.model.save_weights(weights_path)

        # evaluate
        evaluate = self.model.evaluate(x=x_test, y=y_test, verbose=0)
        loss, accuracy = evaluate[:2]

        # save results
        self._save_result(train_uid, description, start_time, fit, loss, accuracy, weights_path)

    def _save_result(self, train_uid, description, start_time, fit, loss, accuracy, weights_path):
        if os.path.isfile(self.train_results_path):
            train_results = utils.json_utils.load(self.train_results_path)
        else:
            train_results = {
                "model": self.__class__.__name__,
                "dataset": os.path.dirname(self.out_folder),
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
            "loss": str(loss),
            "acc": str(accuracy),
            "weights_path": weights_path
        })
        utils.json_utils.dump(train_results, self.train_results_path)

    def load_weights(self, weights_file_path):
        if not os.path.isfile(weights_file_path):
            raise Exception("weights file '{}' not found.".format(weights_file_path))

        self.model.load_weights(weights_file_path)

    def predict(self, x_predict):
        prediction = self.model.predict(x=x_predict, batch_size=128)
        return prediction
