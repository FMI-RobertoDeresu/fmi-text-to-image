import utils
import time
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tf_imports import Model, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten, Dense, Reshape
from tf_imports import TensorBoard, EarlyStopping
from tf_imports import multi_gpu_model


class CAE:
    def __init__(self, input_shape, use_tpu=False, gpus=None):
        if use_tpu and gpus > 0:
            raise Exception("If tpus are used, gpus must be 0.")

        if gpus is not None and gpus < 2:
            raise Exception("If gpus is specified, then it must be and integer >= 2.")

        self.input_shape = input_shape
        self.optimizer = None
        self.loss = None
        self.model = self._create_model(input_shape, use_tpu, gpus, False)
        self.model_compiled = False

    @staticmethod
    def _create_model(input_shape, use_tpu, gpus, print_model_summary):
        # N, M, _ = input_shape
        # input
        input_layer = Input(shape=input_shape)  # (N, M, 1)

        # encoder
        encoder = Conv2D(2, (3, 3), padding='same', activation='relu')(input_layer)  # (N, M, 2)
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

        encoder = Dense(2048)(encoder)  # (2048)
        encoder = Dropout(0.1)(encoder)  # (2048)

        encoder = Dense(1024)(encoder)  # (1024)
        encoder = Dropout(0.1)(encoder)  # (1024)

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

        # model
        model = Model(input_layer, decoder)
        if print_model_summary:
            model.summary()

        # tpu model
        if use_tpu:
            model = tf.contrib.tpu.keras_to_tpu_model(
                model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu='demo-tpu')
                )
            )

        # gpu model
        if (gpus or 0) > 1:
            model = multi_gpu_model(model, gpus=gpus, cpu_relocation=True)

        return model

    def compile(self, optimizer, loss):
        if self.model_compiled:
            raise Exception("The model was already compiled.")

        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model_compiled = True

    def train(self, x_train, y_train, x_test, y_test, batch_size, out_folder):
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
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True)

        # tensor_board_log_dir = Path("{}/tensorboard/{}".format(out_folder, description))
        # tensor_board_log_dir.mkdir(parents=True, exist_ok=True)
        # tensor_board = TensorBoard(log_dir=str(tensor_board_log_dir))

        callbacks = [early_stopping]

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

        # save
        weights_relative_path = Path("weights/{}.h5".format(description))
        weights_path = Path(out_folder, weights_relative_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(weights_path))

        # evaluate
        evaluate = self.model.evaluate(x=x_test, y=y_test, verbose=0)
        loss, accuracy = evaluate[:2]

        # save results
        results_path = Path("{}/results.json".format(out_folder))
        self._save_result(results_path, train_uid, description, start_time, fit, loss, accuracy, weights_relative_path)

    def _save_result(self, path, train_uid, description, start_time, fit, loss, accuracy, weights_path):
        if path.exists():
            train_results = utils.json_utils.load(path)
        else:
            train_results = {
                "model": self.__class__.__name__,
                "input_shape": self.input_shape,
                "training_sessions": []
            }

        train_results["training_sessions"].append({
            "uid": train_uid,
            "description": description,
            "elapsed_time": str(datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)),
            "trained_epochs": len(fit.epoch),
            "loss": str(loss),
            "acc": str(accuracy),
            "weights_path": weights_path.as_posix()
        })
        utils.json_utils.dump(train_results, path)

    def load_weights(self, weights_file_path):
        if not Path(weights_file_path).exists():
            raise Exception("weights file '{}' not found.".format(weights_file_path))

        self.model.load_weights(weights_file_path)

    def predict(self, x_predict):
        prediction = self.model.predict(x=x_predict, batch_size=128)
        return prediction
