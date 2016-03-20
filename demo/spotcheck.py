#!/usr/bin/env python

from sklearn import cross_validation
from sklearn import datasets

import numpy
import arff
from tabulate import tabulate

# Classifiers to spot check
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
from sklearn import naive_bayes

def getLabelValues(labelTypes):
    vals = {}
    count = 0
    for label in labelTypes:
        vals[label] = count
        count += 1
    return vals

def assignValuesToLabels():
    pass

def loadArffDataset(filename, displayData=False):
    with open(filename) as f:
        data = arff.load(f)

    if displayData:
        attributeHeaders = [ attr[0] for attr in data['attributes'] ]
        rows = data['data']
        print tabulate(rows, headers=attributeHeaders)

    # Assign each label a numerical value and construct list
    labelTypes = set([ item[-1] for item in rows ]) # just get label types
    labelValues = assignValuesToLabels(labels)

    # Convert data to format supported by scikit-learn
    inputs = numpy.array([ item[:-1] for item in rows ]) # features
    outputs = numpy.array([ for val in labelValues ])



    assignValuesToClasses()
    labels = numpy.array([ item[-1] for item in rows ])


    numInputFeatures = len(data['attributes'])

    return inputs, labels, numInputFeatures

def evaluateClassifiers(classifiers, inputs, labels, kFolds):
    results = {}
    for name, classifier in classifiers.items():
        results[name] = cross_validation.cross_val_score(
                                          classifier, inputs, labels, cv=kFolds)

if __name__ == '__main__':
    # Load dataset
    inputs, labels, numFeatures = loadArffDataset(
                                     'data/faces_vegetables_dataset.arff', True)

    # Construct all classifiers we wish to test, with 'standard' parameters
    classifiers = {
        'Linear Regression':
            linear_model.LinearRegression(),
        'SVM':
            svm.SVC(kernel='linear', C=1),
        'Decision Tree':
            tree.DecisionTreeClassifier(criterion='gini', splitter='best'),
        #'Feed-Forward NN':
        #    neural_network.MLPClassifier(hidden_layer_sizes=(numFeatures)),
        'Gaussian Naive Bayes':
            naive_bayes.GaussianNB(),
        'Multi-Nomial Naive Bayes':
            naive_bayes.MultinomialNB(),
        'Bernoulli Naive Bayes':
            naive_bayes.BernoulliNB(),
    }

    # Test classifiers and rank them from best performing to least
    results = evaluateClassifiers(classifiers, inputs, labels, 10)
    for name, scores in results.items():
        print name
        print scores
        print
