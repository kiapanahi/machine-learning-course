import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV


def create_blobs(samples):
    centers = [(0, -4), (0, 3)]
    (X, y) = make_blobs(n_samples=samples,
                        # n_features=1,
                        centers=centers,
                        # cluster_std=0.5,
                        random_state=42)
    return (X, y)


def scatter_blobs(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.show()


def fit_and_predict(estimator, X, y):
    estimator.fit(X, y)
    estimator.predict(X)


def grid_search(estimator, attributes, X, y):
    grid_search = GridSearchCV(estimator,
                               attributes,
                               cv=3,
                               scoring='accuracy')

    grid_search.fit(X, y)
    return grid_search


if __name__ == "__main__":
    (X, y) = create_blobs(2000)

    dt_classifier = DecisionTreeClassifier(max_depth=2)
    fit_and_predict(dt_classifier, X, y)

    sgd_classifier = SGDClassifier()
    fit_and_predict(sgd_classifier, X, y)

    dt_classifier_attributes = {
        'max_depth': [2, 4, 6, 8],
        'min_samples_leaf': [1, 5, 10]
    }
    dt_grid_search = grid_search(dt_classifier, dt_classifier_attributes, X, y)
    sgd_grid_search = grid_search(sgd_classifier, {}, X, y)

    print('dt_classifer best estimator: {}'.format(
        dt_grid_search.best_estimator_))
    print('sgd_classifer best estimator: {}'.format(
        sgd_grid_search.best_estimator_))

    export_graphviz(dt_classifier,
                    './out/dt_classifier_export_s06',
                    class_names=['blue', 'yellow'],
                    feature_names=['feat1', 'feat2'],
                    max_depth=2,
                    filled=True,
                    rounded=True)

    scatter_blobs(X, y)
