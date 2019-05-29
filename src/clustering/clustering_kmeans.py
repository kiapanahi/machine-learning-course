import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

blob_centers = np.array([
    [.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]
])
blob_std = np.array([.4, .3, .1, .1, .1])

X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std)


def plot(X, y):
    plt.scatter(X[:, 0], X[:, 1], s=3, alpha=.7, c=y)
    plt.show()


if __name__ == "__main__":
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    print('labels: {}'.format(kmeans.labels_))
    print('inertia: {}'.format(kmeans.inertia_))
    print('centers: {}'.format(kmeans.cluster_centers_))

    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.7, c=y)
    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.7, c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                # s=20,
                marker='x',
                c='r')
    plt.show()
