#!/usr/bin/env python

# Test Harness Tools
from sklearn import cross_validation
# Classifiers to spot check
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
from sklearn import naive_bayes

# Other Useful Dependencies
import numpy
import arff
from tabulate import tabulate
from operator import itemgetter


def createLabelValueMapping(labelTypes):
    vals = {}
    count = 0

    # Sorted given label types to ensure mapping is the same when the same
    # types are given in multiple runs
    for label in sorted(labelTypes):
        vals[label] = count
        count += 1

    return vals

def assignValuesToLabels(rawLabels, labelValueMapping):
    return [ labelValueMapping[l] for l in rawLabels ]

def loadArffDataset(filename, displayData=False):
    with open(filename) as f:
        data = arff.load(f)

    if displayData:
        attributeHeaders = [ attr[0] for attr in data['attributes'] ]
        rows = data['data']
        print tabulate(rows, headers=attributeHeaders)

    # Assign each label a numerical value
    rawLabels = [ item[-1] for item in rows ]
    labelValueMapping = createLabelValueMapping(set(rawLabels))
    # Use mapping to convert labels to a number (for sklearn)
    labelValues = assignValuesToLabels(rawLabels, labelValueMapping)

    # Convert data to format supported by scikit-learn
    inputs = numpy.array([ item[:-1] for item in rows ]) # features
    labels = numpy.array(labelValues)
    numInputFeatures = len(data['attributes'])

    return inputs, labels, numInputFeatures

def evaluateClassifiers(classifiers, inputs, labels, kFolds):
    results = {}
    for name, classifier in classifiers.items():
        results[name] = cross_validation.cross_val_score(
                                          classifier, inputs, labels, cv=kFolds)
    return results

def computeOverallScores(results):
    overallScores = []
    for clsName, scores in results.items():
        mean = scores.mean()
        confidenceInterval = scores.std() * 2
        worstCase = mean - confidenceInterval
        overallScores.append( [clsName, mean, confidenceInterval, worstCase] )

    # Sort by mean score descending before returning
    overallScores.sort(key=itemgetter(1), reverse=True)
    return overallScores

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

    # Test classifiers and compute their mean scores
    results = evaluateClassifiers(classifiers, inputs, labels, 10)
    scores = computeOverallScores(results)

    # Output scores in tabular format
    # Note that the overall scores list is already is sorted from highest mean
    # score to lowest
    print tabulate(
        scores,
        headers=['Classifier', 'Mean Acc.', 'Conf. Interval', 'Worst Acc.'])
