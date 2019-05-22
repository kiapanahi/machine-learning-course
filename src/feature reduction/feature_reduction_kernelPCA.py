import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

if __name__ == "__main__":

    x, y = make_swiss_roll(1000, 0.0, 42)

    x_tr, x_ts, y_tr, y_ts = x[:900, :], x[900:, :], y[:900], y[900:]

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=.04)

    kpca.fit(x_tr)

    x_result = kpca.transform(x_tr)

    plt.scatter(x_result[:, 0], x_result[:, 1], c=y_tr)
    plt.show()
