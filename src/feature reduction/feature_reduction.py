import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA


def plot_number(x):
    x = x.reshape(28, 28)
    plt.imshow(x)
    plt.show()


if __name__ == "__main__":

    mnist = fetch_mldata("MNIST original")
    x = mnist.data
    y = mnist.target

    # plot_number(x[36000])

    x_tr, x_ts, y_tr, y_ts = x[:60000, :], x[60000:, :], y[:60000], y[60000:]

    # x_center = x_tr - np.mean(x_tr)
    # u, s, v = np.linalg.svd(x_center)

    # w = v.T[:, :100]
    # x_tr_red = x_tr@w

    # pca = PCA(n_components=100)
    # x_tr_red_2 = pca.fit_transform(x_tr)
    # # print(pca.components_)

    pca2 = PCA(n_components=.95)
    pca2.fit_transform(x_tr)
    x_tr_red = pca2.transform(x_tr)
    print("shape is: {}".format(x_tr_red.shape))
    x_tr_recon = pca2.inverse_transform(x_tr_red)
    print("reconstructed: {}".format(x_tr_recon.shape))
    plot_number(x_tr_recon[36000])
