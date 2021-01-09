#include "SOM.cu"
#include "training_metrics.cu"
#include "crossvalidation.cu"
#include "io.cu"

#include <algorithm>
#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include "constants.cuh"

SOM::SelfOrganizingMap trainSingleSOM(SOM::configuration conf, digitSet inputSet);
std::vector<std::pair<double, int>> classificationAccuracies(SOM::configuration conf, digitSet train, digitSet test);

void printAccuracyStatistics(std::vector<std::pair<double, int>> &accuracies_and_size);

int executeWithOptions(io::argumentOptions options) {
    SOM::configuration conf = io::SOM::parse_SOM_configuration(options);
    digitSet input = io::SOM::parseInputSet(options, conf);

    if (!options.count(OPTION_FLAG_PATH_TEST_INPUT)) {
        SOM::SelfOrganizingMap som = trainSingleSOM(conf, input);
        std::string outputName = options.count(OPTION_FLAG_PATH_SOM_OUTPUT) > 0
                                 ? options[OPTION_FLAG_PATH_SOM_OUTPUT]
                                 : DEFAULT_SOM_FILENAME;
        std::cout << "[SOM] saving final SOM to   " << outputName << std::endl;
        io::SOM::writeSOM(outputName, som);
    } else {
        digitSet test = io::SOM::parseTestingSet(options, conf);
        std::vector<std::pair<double, int>> accuracies_and_size = classificationAccuracies(conf, input, test);
        printAccuracyStatistics(accuracies_and_size);
    }
    return 0;
}

void printAccuracyStatistics(std::vector<std::pair<double, int>> &accuracies_and_size) {
    std::vector<double> accuracies;
    std::for_each(accuracies_and_size.begin(), accuracies_and_size.end(),
                  [&accuracies](std::pair<double, int> p) { accuracies.push_back(p.first); });
    std::cout << "[crossvalidate] accuracies_and_size\n";
    std::for_each(accuracies_and_size.begin(), accuracies_and_size.end(),
                  [](std::pair<double, int> accuracyp) {
                      std::cout << accuracyp.first << " +-" << 1.96 *
                                                               (sqrt(accuracyp.first *
                                                                     (1 -
                                                                      accuracyp.first) /
                                                                     accuracyp.second))
                                << " with 95% confidence" << std::endl;
                  });
    std::cout << std::endl;
    double sum = 0;
    std::for_each(accuracies_and_size.begin(), accuracies_and_size.end(),
                  [&sum](std::pair<double, int> accuracyp) { sum += accuracyp.first; });
    std::pair<double, double> accuracy_confidence = training_metrics::confidence95(accuracies);
    if (accuracies_and_size.size() != 0) {
        std::cout << "Accuracy with 95% confidence " << accuracy_confidence.first << " +-"
                  << accuracy_confidence.second << std::endl;
    }
}

SOM::SelfOrganizingMap trainSingleSOM(SOM::configuration conf, digitSet inputSet) {
    SOM::SelfOrganizingMap som(inputSet, conf.mapW, conf.mapH);
    if (conf.animation) {
        som.train(conf.maxT, [&](int T, int maxT, SOM::SelfOrganizingMap *map) {
            if ((T + maxT / conf.frameCount) % (maxT / conf.frameCount) == 0) {
                std::ofstream file;
                file.open(
                        conf.animationPath + "/" + (std::to_string(T / (maxT / conf.frameCount)) + ".som").c_str());
                map->printMapToStream(file);
                file.close();
            }
        });
    } else {
        som.train(conf.maxT);
    }
    return som;
}

double sampleClassificationAccuracy(SOM::configuration conf,
                                    digitSet &splitTrain, digitSet &splitTest) {
    std::vector<digitSet> filtered(conf.classCount, digitSet(conf.digitW, conf.digitH));
    for (int i = 0; i < conf.classCount; i++) {
        filtered[i] = digitSet::filterByValue(splitTrain, i);
    }
    std::vector<SOM::SelfOrganizingMap> maps;
    for (int i = 0; i < conf.classCount; i++) {
        maps.push_back(SOM::SelfOrganizingMap(filtered[i], conf.mapW, conf.mapH));
        maps[i].train(conf.maxT, [&](int T, int maxT, SOM::SelfOrganizingMap *map) {
            if ((T + maxT / conf.frameCount) % (maxT / conf.frameCount) == 0) {
                std::ofstream file;
                file.open(conf.animationPath + "/" +
                          std::to_string(i) + "_" + (std::to_string(T / (maxT / conf.frameCount)) +
                                                     ".som").c_str());
                map->printMapToStream(file);
                file.close();
            }
        });
        maps[i].train(conf.maxT);
    }
    splitTest.minMaxFeatureScale();
    training_metrics::confusion_matrix confusionMatrix(conf.classCount, std::vector<int>(conf.classCount, 0));
    std::for_each(splitTest.getDigits().begin(), splitTest.getDigits().end(),
                  [&](auto &sample) { ++confusionMatrix[SOM::classify(maps, sample)][sample.getValue()]; });
    training_metrics::print_confusion_matrix(std::cout, confusionMatrix, conf.classCount);
    double accuracy = training_metrics::accuracy(confusionMatrix, conf.classCount);

    // Precision
    std::vector<double> positive_precisions = training_metrics::positive_precisions(confusionMatrix, conf.classCount);
    std::cout << "Precisions" << std::endl;
    std::for_each(positive_precisions.begin(), positive_precisions.end(), [](double pr) { std::cout << " " << pr; });
    std::cout << std::endl;

    // Sensitivity
    std::vector<double> sensitivities = training_metrics::sensitivity(confusionMatrix, conf.classCount);
    std::cout << "Sensitivities" << std::endl;
    std::for_each(sensitivities.begin(), sensitivities.end(), [](double se) { std::cout << " " << se; });
    std::cout << std::endl;

    // Fscore
    std::vector<double> fscores = training_metrics::fscore(positive_precisions, sensitivities);
    std::cout << "Fscores" << std::endl;
    std::for_each(fscores.begin(), fscores.end(), [](double fs) { std::cout << " " << fs; });
    std::cout << std::endl;

    // AUC
    std::vector<double> areas = training_metrics::AUC(confusionMatrix, conf.classCount);
    std::cout << "AUC" << std::endl;
    std::for_each(areas.begin(), areas.end(), [](double area) { std::cout << " " << area; });
    std::cout << std::endl;
    return accuracy;
}

std::vector<std::pair<double, int>> classificationAccuracies(SOM::configuration conf, digitSet train, digitSet test) {
    conf.printConfiguration(std::cout);;
    std::vector<std::pair<double, int>> accuracies;
    int k = DEAFULT_COUNT_CROSS_VALIDATION_K;
    for (int i = 0; i < k; ++i) {
        std::pair<digitSet, digitSet> splitData = modelValidation::crossvalidate_split(train, test, k, i);
        accuracies.push_back(std::make_pair(
                sampleClassificationAccuracy(conf, splitData.first, splitData.second),
                splitData.second.getDigits().size()));
    }
    return accuracies;
}