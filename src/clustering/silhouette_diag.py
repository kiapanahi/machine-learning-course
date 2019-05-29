import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from sklearn.metrics import silhouette_samples, silhouette_score


def silhouette_diag(k, kmeans, X):

    y_pred = kmeans.labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.9)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.ylabel("Cluster")
    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Silhouette Coefficient")
    silhouette_score = silhouette_score(X, kmeans.labels_)
    plt.axvline(x=silhouette_score, color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)
    plt.show()
