#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "digitSet.cu"
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

std::pair<digitSet, digitSet> random_split(const digitSet& train,
    const digitSet& test)
{
    if(train.getDigits().size() == 0 || test.getDigits().size() == 0) {
        std::cerr<<"[crossvalidation::random_split] error, train or test empty";
        exit(1);
    }

    digitSet splitTrain(train.getDigits()[0].getWidth(),train.getDigits()[0].getHeight());
    digitSet splitTest(train.getDigits()[0].getWidth(),train.getDigits()[0].getHeight());
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

std::pair<digitSet, digitSet> crossvalidate_split(const digitSet& train,
    const digitSet& test, int k,
    int testFractionIndex)
{
    if(train.getDigits().size() == 0 || test.getDigits().size() == 0) {
        std::cerr<<"[crossvalidation::random_split] error, train or test empty";
        exit(1);
    }

    digitSet splitTrain(train.getDigits()[0].getWidth(),train.getDigits()[0].getHeight());
    digitSet splitTest(train.getDigits()[0].getWidth(),train.getDigits()[0].getHeight());
    
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
#endif