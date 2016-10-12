#!/usr/bin/env python

import autosklearn.classification
from sklearn import preprocessing
import numpy as np
import arff
from tabulate import tabulate

# ------------------------------------------------------------------------------
# Dataset Loading
# ------------------------------------------------------------------------------
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


def loadArffDataset(filename, normalise, displayData=False):
    with open(filename) as f:
        data = arff.load(f)

    if displayData:
        attributeHeaders = [ attr[0] for attr in data['attributes'] ]
        rows = data['data']
        print tabulate(rows, headers=attributeHeaders)

    # Assign each label a numerical value
    # (required by most classifiers)
    rawLabels = [ item[-1] for item in data['data'] ]
    labelValueMapping = createLabelValueMapping(set(rawLabels))
    # Use mapping to convert labels to a number (for sklearn)
    labelValues = assignValuesToLabels(rawLabels, labelValueMapping)

    # Structure input/label data in a format sklearn understands
    featureVecs = np.array([ item[:-1] for item in rows ]) # features
    labels = np.array(labelValues)
    numInputFeatures = len(data['attributes'])
    numLabelTypes = len(labelValueMapping)

    if normalise:
        featureVecs = preprocessing.normalize(featureVecs)

    return featureVecs, labels, numInputFeatures, numLabelTypes


def copyAndShuffle(arr):
    randomisedArr = np.copy(arr)
    np.random.shuffle(randomisedArr)
    return randomisedArr


def splitDataset(featureVecs, labels, percentInTraining):
    randomisedFeatureVecs = copyAndShuffle(featureVecs)
    randomisedLabels = copyAndShuffle(labels)

    numInTraining = int(len(randomisedFeatureVecs) * percentInTraining)

    X_train = randomisedFeatureVecs[:numInTraining]
    y_train = randomisedLabels[:numInTraining]
    X_test = randomisedFeatureVecs[numInTraining:]
    y_test = randomisedLabels[numInTraining:]

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------------------
# Test Harness
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load test dataset
    featureVecs, labels, numFeatures, numLabelTypes = loadArffDataset(
                                           'data/faces_vegetables_dataset.arff',
                                            normalise=True,
                                            displayData=True)

    # Split input data into train and test datasets
    X_train, y_train, X_test, y_test = splitDataset(featureVecs, labels, 0.50)

    # Train classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120)
    automl.fit(X_train, y_train)

    # Output accuracy of classifier
    print automl.score(X_test, y_test)

    # Output structure of classifier
    print automl.show_models()
