#!/usr/bin/env python

import autosklearn.classification
import sklearn.datasets
import numpy as np

# URL explaining dataset used in this example:
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

if __name__ == '__main__':
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1000:]
    y_test = y[1000:]

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30)
    automl.fit(X_train, y_train)

    print "Score:", automl.score(X_test,y_test)
    print automl.show_models()
