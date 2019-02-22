import numpy as np
import tensorflow as tf


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
    # placeholders
    batch_size_ = tf.placeholder(tf.int64)
    features_ = tf.placeholder(tf.float32, shape=[None, 2])
    labels_ = tf.placeholder(tf.float32, shape=[None, 1])

    # datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((features_, labels_)).batch(batch_size_).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((features_, labels_)).batch(batch_size_)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iter.get_next()

    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    # make model
    encoded, decoded, logits = create_model()

    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add noise to input
    noisy_train_data = train_data + noise_factor * np.random.randn(*train_data.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op, feed_dict={features_: noisy_train_data, labels_: train_data, batch_size_: batch_size})
        for epoch in range(epochs):
            num_batches = train_data.shape[0] // batch_size
            loss = 0
            for _ in range(num_batches):
                batch_loss, opt_batch = sess.run([loss, opt], feed_dict={features_: noisy_train_data, labels_: train_data})
                loss += batch_loss
        print("Epoch: {}/{}...".format(epoch + 1, epochs), "Training loss: {:.4f}".format(loss))


def mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train(x_train, x_test, epochs=3)


if __name__ == "__main__":
    mnist()
