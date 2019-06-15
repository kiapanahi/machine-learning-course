import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, SGDClassifier


def plot_sample(X, sample_number: int):
    sample_image = X[sample_number, :].reshape(8, 8)
    plt.imshow(sample_image)
    plt.show()


def get_test_train_set(X: np.ndarray, y: np.ndarray):

    size = y.size
    tr_size = int(size*.8)
    shuffle_map = np.random.permutation(tr_size)

    X_tr = X[:tr_size, :]
    X_tst = X[tr_size:, :]
    y_tr = y[:tr_size]
    y_tst = y[tr_size:]

    X_tr, y_tr = X_tr[shuffle_map], y_tr[shuffle_map]

    return (X_tr, X_tst, y_tr, y_tst)


if __name__ == "__main__":
    (X, y) = load_digits(return_X_y=True)

    (X_tr, X_tst, y_tr, y_tst) = get_test_train_set(X, y)

    logistic_regression_clf = LogisticRegression(
        multi_class='auto',
        random_state=42
    )
    logistic_regression_clf.fit(X_tr, y_tr)
    X_prd = logistic_regression_clf.predict(X_tst)

    for i in range(y_tst.size):
        print('{}<->{}'.format(X_prd[i], y_tst[i]))
        if X_prd[i] != y_tst[i]:
            print('prediction was wrong!')

    print(logistic_regression_clf.predict_proba(X_tst))
    print(logistic_regression_clf.classes_)
    print(logistic_regression_clf.coef_)
    print(logistic_regression_clf.intercept_)
