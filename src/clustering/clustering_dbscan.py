from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=2000, random_state=42, noise=.1)

dbscan = DBSCAN(eps=.1, min_samples=5)
dbscan.fit(X)
print('label: {}'.format(dbscan.labels_))

outliers = X[dbscan.labels_ == -1]
non_outliers = X[dbscan.labels_ > -1]

plt.scatter(non_outliers[:, 0],
            non_outliers[:, 1],
            c=dbscan.labels_[dbscan.labels_ > -1],
            alpha=.6)

plt.scatter(outliers[:, 0],
            outliers[:, 1],
            marker='x',
            
            s=50,
            c='k',)
plt.show()
