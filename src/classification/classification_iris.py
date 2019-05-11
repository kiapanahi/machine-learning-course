import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets.base import get_data_home
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = iris['data']
Y = iris['target']

X_tr, X_test = X[:120, :], X[120:, :]
Y_tr, Y_test = Y[:120], Y[:120]

permutation_indice = np.random.permutation(120)
X_tr, Y_tr = X_tr[permutation_indice], Y_tr[permutation_indice]

dt_classifier = DecisionTreeClassifier()

X_tr = X_tr[:, 2:]
X_test = X_test[:, 2:]

dt_classifier.fit(X_tr, Y_tr)

y_predict = dt_classifier.predict(X_test)

export_graphviz(dt_classifier,
                './out/dt_classifier_export',
                class_names=iris['target_names'],
                feature_names=iris['feature_names'][2:],
                max_depth=2,
                filled=True,
                rounded=True)


# plt.scatter(X[:, 2], X[:, 3], c=Y, cmap=plt.get_cmap('jet'))
# plt.xlabel('Petal length (cm)')
# plt.ylabel('Petal width (cm)')
# x = np.arange(0, 8, 0.1)
# y = np.arange(0, 2.5, 0.05)
# line1 = 2.45*np.ones(len(y))
# line2 = 1.65*np.ones(len(x))
# plt.plot(line1, y)
# plt.plot(x, line2)
# plt.show()

attributes = {'max_depth': [2, 4, 6, 8], 'min_samples_leaf': [1, 5, 10]}
grid_search = GridSearchCV(dt_classifier, attributes, cv=3, scoring='accuracy')
grid_search.fit(X_tr, Y_tr)

print(grid_search.best_estimator_)
