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

    fig_size = [6, 8]
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    for index, img in enumerate(imgs):
        axi = ax.flat[index]
        axi.imshow(img)

        title = "[{},{}]".format(index // n_cols, index % n_cols)
        if labels is not None:
            title = "{} - {}".format(title, labels[index])
        axi.set_title(title)

    plt.tight_layout(True)
    plt.title(title)

    figure = plt.figure()
    plt.show()

    if save_path is not None:
        figure.savefig(save_path)
