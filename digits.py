import numpy as np
import tensorflow as tf

def create_model():
    learning_rate = 0.001
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 14x14x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 7x7x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 4x4x16


    ### Decoder
    upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x16
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 14x14x16
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 28x28x32
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32
    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
    #Now 28x28x1
    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)
    # Pass logits through sigmoid and calculate the cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return (inputs_, targets_, cost, opt)


def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    inputs_, targets_, cost, opt = create_model()

    sess = tf.Session()
    epochs = 1
    batch_size = 200
    # Set's how much noise we're adding to the MNIST images
    noise_factor = 0.5
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(x_train.shape(0) // batch_size):
            batch = mnist.train.next_batch(batch_size)
            # Get images from the batch
            imgs = batch[0].reshape((-1, 28, 28, 1))

            # Add random noise to the input images
            noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            # Noisy images as inputs, original images as targets
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                             targets_: imgs})
    print("Epoch: {}/{}...".format(e + 1, epochs),
          "Training loss: {:.4f}".format(batch_cost))

if __name__ == "__main__":
    mnist()
