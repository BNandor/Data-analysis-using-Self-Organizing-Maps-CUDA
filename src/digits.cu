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

training_metrics::Metrics classificationMetrics(SOM::configuration conf, digitSet train, digitSet test);

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
        training_metrics::Metrics metrics = classificationMetrics(conf, input, test);
        metrics.printMetricsWithCIs();
    }
    return 0;
}

SOM::SelfOrganizingMap trainSingleSOM(SOM::configuration conf, digitSet inputSet) {
    SOM::SelfOrganizingMap som(inputSet, conf);
    if (conf.animation) {
        som.train(conf.maxT, [&](int T, SOM::SelfOrganizingMap *map) {
            if ((T + map->getConf().maxT / map->getConf().frameCount) %
                (map->getConf().maxT / map->getConf().frameCount) == 0) {
                std::string output = map->getConf().animationPath + "/" +
                                     std::to_string(T / (map->getConf().maxT / map->getConf().frameCount)) + ".som";
                io::SOM::printIntermediateSOM(T, map, output);
            }
        });
    } else {
        som.train(conf.maxT);
    }
    return som;
}

void sampleClassificationMetrics(SOM::configuration conf,
                                 digitSet &splitTrain, digitSet &splitTest, training_metrics::Metrics &metrics) {
    std::vector<digitSet> filtered(conf.classCount, digitSet(conf.digitW, conf.digitH));
    for (int i = 0; i < conf.classCount; i++) {
        filtered[i] = digitSet::filterByValue(splitTrain, i);
    }
    std::vector<SOM::SelfOrganizingMap> maps;
    for (int i = 0; i < conf.classCount; i++) {
        maps.push_back(SOM::SelfOrganizingMap(filtered[i], conf));
        maps[i].train(conf.maxT, [&](int T, SOM::SelfOrganizingMap *map) {
            if ((T + map->getConf().maxT / map->getConf().frameCount) %
                (map->getConf().maxT / map->getConf().frameCount) == 0) {
                std::string output = map->getConf().animationPath + "/" +
                                     std::to_string(i) + "_" +
                                     std::to_string(T / (map->getConf().maxT / map->getConf().frameCount)) +
                                     ".som";
                io::SOM::printIntermediateSOM(T, map, output);
            }
        });
        maps[i].train(conf.maxT);
    }

    splitTest.minMaxFeatureScale();
    training_metrics::confusion_matrix confusionMatrix(conf.classCount, std::vector<int>(conf.classCount, 0));

    std::for_each(splitTest.getDigits().begin(), splitTest.getDigits().end(),
                  [&](auto &sample) { ++confusionMatrix[SOM::classify(maps, sample)][sample.getValue()]; });
    training_metrics::print_confusion_matrix(std::cout, confusionMatrix, conf.classCount);
    metrics.add_matrix(confusionMatrix);
}

training_metrics::Metrics classificationMetrics(SOM::configuration conf, digitSet train, digitSet test) {
    conf.printConfiguration(std::cout);
    std::vector<std::pair<double, int>> accuracies;
    int k = DEAFULT_COUNT_CROSS_VALIDATION_K;
    training_metrics::Metrics metrics;
    for (int i = 0; i < k; ++i) {
        std::pair<digitSet, digitSet> splitData = modelValidation::crossvalidate_split(train, test, k, i);
        sampleClassificationMetrics(conf, splitData.first, splitData.second, metrics);
    }
    return metrics;
}