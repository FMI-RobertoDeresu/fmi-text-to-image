import numpy as np
import utils
import const
from tf_imports import tf, Callback
from models.word2vec import Word2Vec


class OutputCheckpoint(Callback):
    def __init__(self, tensor_board_writer, inputs, print_every=1):
        super().__init__()
        self.writer = tensor_board_writer
        self.inputs = inputs
        self.print_every = print_every

        inputs_word2vec = np.array(Word2Vec.get_instance().get_embeddings_remote(inputs))
        inputs_word2vec = utils.process_w2v_inputs(inputs_word2vec, const.INPUT_SHAPE)

        self.inputs = inputs
        self.inputs_word2vec = inputs_word2vec

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every > 0:
            return

        outputs = self.model.predict(x=self.inputs_word2vec, batch_size=128)

        name = "output checkpoint epoch {}".format(epoch + 1)
        summary = tf.summary.image(name=name, tensor=outputs, max_outputs=4)
        with tf.Session().as_default():
            self.writer.add_summary(summary.eval(), global_step=epoch)
        self.writer.flush()
