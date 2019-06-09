def plot_boundaries(clf, axes):
    # take a classifier clf
    # plot its boundaries based on the axes
    # axes = [x_min,x_max,y_min,y_max]

    import numpy as np
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    import matplotlib.pyplot as plt
    plt.contourf(x0, x1, y_pred, alpha=0.3, cmap=plt.get_cmap('jet'))
