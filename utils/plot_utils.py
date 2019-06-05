import matplotlib.pyplot as plt
import numpy as np


def plot_image(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def plot_multiple_images(imgs, title=None, labels=None, save_path=None):
    n_imgs = len(imgs)
    n_rows = int(np.round(np.sqrt(n_imgs)))
    n_cols = int(np.ceil(n_imgs / n_rows))

    fig_size = [8, 8]
    fig = plt.figure(figsize=fig_size)
    plt.title(title, y=0.9)
    plt.axis("off")

    for index, img in enumerate(imgs):
        ax = fig.add_subplot(n_rows, n_cols, index + 1)
        ax.imshow(img)
        label = "[{},{}]".format(index // n_cols, index % n_cols)
        if labels is not None:
            label = "{} - {}".format(label, labels[index])
        ax.set_title(label)

    plt.show()

    if save_path is not None:
        fig.savefig(save_path)

    plt.close()
