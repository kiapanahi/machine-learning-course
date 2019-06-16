from gplearn.genetic import SymbolicRegressor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(-1, 1, .1)
x2 = np.arange(-1, 1, .1)

x1, x2 = np.meshgrid(x1, x2)

y = x1**2 - x2**2 + x2 - 1

ax = plt.figure(figsize=[8, 5]).add_subplot(111, projection='3d')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

ax.plot_surface(x1, x2, y, alpha=0.8)

X_tr = np.random.uniform(-1, 1, [50, 2])
X_ts = np.random.uniform(-1, 1, [50, 2])

y_tr = X_tr[:, 0]**2 - X_tr[:, 1]**2 + X_tr[:, 1] - 1
y_ts = X_ts[:, 0]**2 - X_ts[:, 1]**2 + X_ts[:, 1] - 1


ax.scatter(X_tr[:, 0], X_tr[:, 1], y_tr)
ax.scatter(X_ts[:, 0], X_ts[:, 1], y_ts)


gp_reg = SymbolicRegressor(population_size=5000,
                           p_crossover=.7,
                           p_subtree_mutation=.1,
                           p_hoist_mutation=.01,
                           p_point_mutation=.1,
                           parsimony_coefficient=.01,
                           stopping_criteria=.01,
                           max_samples=.9,
                           verbose=True)

gp_reg.fit(X_tr, y_tr)

print(gp_reg._program)
print(gp_reg.run_details_)
