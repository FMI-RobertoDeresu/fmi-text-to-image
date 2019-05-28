import pathlib
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class CAE:
    def __init__(self, dataset_name):
        self.checkpoint_dir = "models/data/cae/{dataset}/"
        self.checkpoint_filepath = self.checkpoint_dir + "checkpoint-{{epoch:02d}}-{{val_acc:.2f}}.hdf5"

        self.tensor_board_log_dir = '/tmp/cae/tensor_board/{dataset}'

        # input
        input_layer = Input(shape=(30, 300))

        # encoder
        encoder = Conv2D(2, (3, 3), activation='relu', padding='same')(input_layer)
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

        self.autoencoder = Model(input_layer, decoder)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, x_train, y_train, dataset):
        # checkpoint
        checkpoint_filepath = self.checkpoint_filepath.format(dataset=dataset)
        pathlib.Path(checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        # early stopping
        early_stopping = EarlyStopping(monitor='val_acc',
                                       min_delta=0.01,
                                       patience=5,
                                       verbose=1,
                                       mode='max',
                                       restore_best_weights=True)

        # tensor board
        tensor_board_log_dir = self.tensor_board_log_dir.format(dataset=dataset)
        pathlib.Path(tensor_board_log_dir).mkdir(parents=True, exist_ok=True)
        tensor_board = TensorBoard(log_dir=tensor_board_log_dir)

        # fit
        callbacks = [checkpoint, tensor_board, early_stopping]
        self.autoencoder.fit(x_train,
                             y_train,
                             epochs=50,
                             batch_size=128,
                             shuffle=True,
                             validation_split=0.2,
                             callbacks=callbacks)
