import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets.base import get_data_home
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
X = iris.data
Y = iris.target

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
                class_names=iris.target_names,
                feature_names=iris.feature_names[2:],
                filled=True,
                rounded=True)
                
print(y_predict)
