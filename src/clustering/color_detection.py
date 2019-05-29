from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

fig = imread('./src/clustering/ladybug.png')

print(fig.shape)
X = fig.reshape(-1, 3)
kmeans = KMeans(n_clusters=8)
kmeans.fit(X)

X_seg = X
for k in range(8):
    X_seg[kmeans.labels_ == k] = kmeans.cluster_centers_[k, :]

fig_seg = X_seg.reshape(fig.shape)
plt.imshow(fig_seg)
plt.show()
