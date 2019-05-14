from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt


def create_blobs(samples):
    centers = [(0, -2), (0, 2)]
    (X, y) = make_blobs(n_samples=samples,
                        # n_features=1,
                        centers=centers,
                        # cluster_std=0.5,
                        random_state=42)
    return (X, y)


def scatter_blobs(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


if __name__ == "__main__":
    (X, y) = create_blobs(2000)
    scatter_blobs(X, y)
