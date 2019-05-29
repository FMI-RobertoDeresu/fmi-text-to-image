import pathlib
import time
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class CAE:
    def __init__(self):
        self.tensor_board_log_dir = 'tensorboard/cae/{dataset}'
        self.checkpoint_dir = "checkpoints/cae/{{dataset}}/{}/".format(int(time.time()))
        self.checkpoint_filepath_template = self.checkpoint_dir + \
                                            "checkpoint-{{epoch:02d}}-{{val_loss:.2f}}-{{val_acc:.2f}}.hdf5"
        self.best_weights_filepath = self.checkpoint_dir + "best_weights.hdf5"

        self._create_model()

    def _create_model(self):
        # input
        input_layer = Input(shape=(30, 300, 1))

        # encoder
        encoder = Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=input_layer.shape)(input_layer)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (50, 150, )

        encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (25, 75, )

        encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (13, 38, )

        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (7, 19, )

        encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (4, 10, )

        encoder = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (2, 5, )

        encoder = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (1, 3, )

        encoder = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (1, 2, )

        encoder = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)  # (1, 1, )

        # decoder
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (2, 2, )

        decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (4, 4, )

        decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (8, 8, )

        decoder = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (16, 16, )

        decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (32, 32, )

        decoder = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (64, 64, )

        decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)  # (128, 128, )

        decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)

        self.model = Model(input_layer, decoder)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train(self, x_train, y_train, dataset):
        # checkpoint
        checkpoint_filepath = self.checkpoint_filepath_template.format(dataset=dataset)
        pathlib.Path(checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')

        # checkpoint
        best_weights_filepath = self.best_weights_filepath.format(dataset=dataset)
        pathlib.Path(best_weights_filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint_best = ModelCheckpoint(filepath=best_weights_filepath,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='min')

        # early stopping
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.01,
                                       patience=5,
                                       verbose=1,
                                       mode='min',
                                       restore_best_weights=True)

        # tensor board
        tensor_board_log_dir = self.tensor_board_log_dir.format(dataset=dataset)
        pathlib.Path(tensor_board_log_dir).mkdir(parents=True, exist_ok=True)
        tensor_board = TensorBoard(log_dir=tensor_board_log_dir)

        # fit
        callbacks = [checkpoint, checkpoint_best, tensor_board, early_stopping]
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=128,
                       epochs=100,
                       verbose=2,
                       shuffle=True,
                       callbacks=callbacks,
                       validation_split=0.1)
