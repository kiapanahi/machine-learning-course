import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

import veronoi as vr

blob_centers = np.array([
    [.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]
])
blob_std = np.array([.4, .3, .1, .1, .1])

X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std)

if __name__ == "__main__":

    # plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.7, c=y)

    kmeans_inertia = []
    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        silhouette = silhouette_score(X, kmeans.labels_)

        print('labels: {}'.format(kmeans.labels_))
        print('inertia: {}'.format(kmeans.inertia_))
        print('centers: {}'.format(kmeans.cluster_centers_))
        kmeans_inertia.append([i, silhouette, kmeans.inertia_])

    kmeans_inertia = np.c_[kmeans_inertia]
    print(kmeans_inertia)
    plt.plot(kmeans_inertia[:, 0], kmeans_inertia[:, 1])
    plt.show()

    # vr.plot_decision_boundaries(kmeans, X,
    #                             resolution=1000,
    #                             show_centroids=True,
    #                             show_xlabels=True,
    #                             show_ylabels=True)

    # plt.show()

    # plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.7, c=y)
    # plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.7, c=kmeans.labels_)
    # plt.scatter(kmeans.cluster_centers_[:, 0],
    #             kmeans.cluster_centers_[:, 1],
    #             # s=20,
    #             marker='x',
    #             c='r')
    # plt.show()
