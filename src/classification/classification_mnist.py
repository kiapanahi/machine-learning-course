from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.base import get_data_home
from sklearn.base import BaseEstimator


class Not5(BaseEstimator):

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros([len(X), 1])


print(get_data_home())
mnist = fetch_mldata("MNIST original")

X = mnist.data
Y = mnist.target

X_tr = X[:60000, :]
X_test = X[60000:, :]
Y_tr = Y[:60000]
Y_test = Y[60000:]

permutation_indice = np.random.permutation(60000)
X_tr, Y_tr = X_tr[permutation_indice], Y_tr[permutation_indice]


Y_tr_5 = (Y_tr == 5)
Y_test_5 = (Y_test == 5)

sgd_classifier = SGDClassifier(random_state=42, max_iter=100)
y_prediction = cross_val_predict(sgd_classifier,
                                 X_tr,
                                 Y_tr_5, cv=3, method='predict')

print(precision_score(Y_tr_5, y_prediction))
print(recall_score(Y_tr_5, y_prediction))

sgd_classifier.fit(X_tr, Y_tr_5)
