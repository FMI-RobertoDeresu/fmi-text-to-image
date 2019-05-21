import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')

FLAGS = flags.FLAGS


def create_model(features):
    # encoder
    conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 28x28x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # now 14x14x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 14x14x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # now 7x7x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 7x7x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
    # now 4x4x16

    # decoder
    upsample1 = tf.image.resize_images(encoded, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # now 7x7x16
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 7x7x16
    upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # now 14x14x16
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 14x14x32
    upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # now 28x28x32
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # now 28x28x32
    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
    # now 28x28x1

    # get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    return encoded, decoded, logits


def train(train_data, test_data, learning_rate=0.001, epochs=100, batch_size=32, noise_factor=0.2):
    input_shape = [None] + list(train_data.shape)[1:4]

    print("Create placeholders")
    batch_size_ = tf.placeholder(tf.int64)
    features_ = tf.placeholder(tf.float32, shape=input_shape)
    labels_ = tf.placeholder(tf.float32, shape=input_shape)

    print("Create datasets")
    train_dataset = tf.data.Dataset.from_tensor_slices((features_, labels_)).batch(batch_size_).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((features_, labels_)).batch(batch_size_)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iter.get_next()

    print("Create iterator init operations")
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    print('Create model')
    encoded, decoded, logits = create_model(features)

    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print(f'Add noise ({noise_factor}) to input')
    noisy_train_data = train_data + noise_factor * np.random.randn(*train_data.shape)

    # Merge all the summaries
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(f'Saving summaries to: {FLAGS.logdir}')
        train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        sess.run(train_init_op, feed_dict={features_: noisy_train_data, labels_: train_data, batch_size_: batch_size})
        for epoch in range(epochs):
            num_batches = train_data.shape[0] // batch_size
            loss_val = 0
            for _ in range(num_batches):
                loss_batch, opt_batch = sess.run([loss, opt])
                loss_val += loss_batch
            print("Epoch: {}/{}...".format(epoch + 1, epochs), "Training loss: {:.4f}".format(loss_val))


def mnist():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
    train(x_train[0:1000], x_test, epochs=3, noise_factor=0)


if __name__ == "__main__":
    mnist()
