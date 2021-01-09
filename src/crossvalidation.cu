#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "digitSet.cu"
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

/** \brief Contains methods for the validation of models.*/
namespace modelValidation {
/** Splits the whole of a dataset consisting of a training and testing dataset in a random manner into new training and testing parts, 
 * preserving the cardinalities.*/
std::pair<digitSet, digitSet> random_split(const digitSet& train /**< The training part of the dataset.*/,
    const digitSet& test /**< The testing part of the dataset.*/)
{
    if (train.getDigits().size() == 0 || test.getDigits().size() == 0) {
        std::cerr << "[crossvalidation::random_split] error, train or test empty";
        exit(1);
    }

    digitSet splitTrain(train.getDigits()[0].getWidth(), train.getDigits()[0].getHeight());
    digitSet splitTest(train.getDigits()[0].getWidth(), train.getDigits()[0].getHeight());
    std::vector<int> indices;
    int trainSize = train.getDigits().size();
    int testSize = test.getDigits().size();

    for (int i = 0; i < trainSize + testSize; ++i) {
        indices.push_back(i);
    }

    std::random_shuffle(indices.begin(), indices.end());

    for (int i = 0; i < trainSize; ++i) {
        if (indices[i] < trainSize) {
            splitTrain.add(train.getDigit(indices[i]));
        } else {
            splitTrain.add(test.getDigit(indices[i] - trainSize));
        }
    }

    for (int i = trainSize; i < trainSize + testSize; ++i) {
        if (indices[i] < trainSize) {
            splitTest.add(train.getDigit(indices[i]));
        } else {
            splitTest.add(test.getDigit(indices[i] - trainSize));
        }
    }
    return std::make_pair(splitTrain, splitTest);
}

/** Splits the dataset consisting of a training and testing part in a crossvalidating manner using a provided
 * k fold parameter, and the index of the fold to be designated as the testing set. */
std::pair<digitSet, digitSet> crossvalidate_split(const digitSet& train /**< The training part of the dataset.*/,
    const digitSet& test /**< The testing part of the dataset.*/, int k /**< The number of folds of the crossvalidation.*/,
    int testFractionIndex /**< The index of the fold to be used as testing.*/)
{
    if (train.getDigits().size() == 0 || test.getDigits().size() == 0) {
        std::cerr << "[crossvalidation::random_split] error, train or test empty";
        exit(1);
    }

    digitSet splitTrain(train.getDigits()[0].getWidth(), train.getDigits()[0].getHeight());
    digitSet splitTest(train.getDigits()[0].getWidth(), train.getDigits()[0].getHeight());

    std::vector<int> indices;
    int trainSize = train.getDigits().size();
    int testSize = test.getDigits().size();

    if (k <= 0 || k >= trainSize + testSize) {
        std::cerr << "[crossvalidate_split] error, invalid k provided:" << k
                  << std::endl;
        exit(1);
    }

    int testFractionSize = (trainSize + testSize) / k;
    int testStartingIndex = testFractionIndex * testFractionSize;
    int testEndingIndex = (testFractionIndex + 1) * testFractionSize;
    if (testStartingIndex < 0 || testStartingIndex >= trainSize + testSize) {
        std::cerr
            << "[crossvalidate_split] error, invalid testStartingIndex provided:"
            << testStartingIndex << std::endl;
        exit(1);
    }
    if (testEndingIndex < 0 || testEndingIndex > trainSize + testSize) {
        std::cerr
            << "[crossvalidate_split] error, invalid testEndingIndex provided:"
            << testEndingIndex << std::endl;
        exit(1);
    }
    for (int i = 0; i < trainSize + testSize; ++i) {
        indices.push_back(i);
    }

    for (int i = 0; i < testStartingIndex; ++i) {
        if (indices[i] < trainSize) {
            splitTrain.add(train.getDigit(indices[i]));
        } else {
            splitTrain.add(test.getDigit(indices[i] - trainSize));
        }
    }

    for (int i = testStartingIndex; i < testEndingIndex; ++i) {
        if (indices[i] < trainSize) {
            splitTest.add(train.getDigit(indices[i]));
        } else {
            splitTest.add(test.getDigit(indices[i] - trainSize));
        }
    }

    for (int i = testEndingIndex; i < trainSize + testSize; ++i) {
        if (indices[i] < trainSize) {
            splitTrain.add(train.getDigit(indices[i]));
        } else {
            splitTrain.add(test.getDigit(indices[i] - trainSize));
        }
    }

    return std::make_pair(splitTrain, splitTest);
}
}
#endif