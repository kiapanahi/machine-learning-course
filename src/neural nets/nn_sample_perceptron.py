import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    X_tr = X[:, 2:]
    y = (y == 0)  # separate classes based on class 0 or NOT! (1,2)
    y = y.astype(int)

    p = Perceptron()
    p.fit(X_tr, y)
    p.predict(X_tr)
    print(p)

    plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y, cmap=plt.get_cmap('jet'))

    import plot_boundaries as pltb

    pltb.plot_boundaries(p, [0, 8, 0, 2.5])
    plt.show()
