#include "SOM.cu"
#include "crossvalidation.cu"
#include "io.cu"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <map>
#include <utility>
#include <vector>

SOM::SelfOrganizingMap trainSingleSOM(io::argumentOptions options);
std::vector<double> classificationAccuracies(io::argumentOptions options);

int main(int argc, const char* argv[])
{
    io::argumentOptions options = io::parse_options(argc, argv);

    if (!options.count("test")) {
        SOM::SelfOrganizingMap som = trainSingleSOM(options);
        std::ofstream output;
        std::string outputName = options.count("outputPath") > 0
            ? options["outputPath"]
            : OUTPUT_SOM_FILENAME;
        std::cout << "[SOM] saving final SOM to   " << outputName << std::endl;
        output.open(outputName);
        som.printMapToStream(output);
        output.close();
    } else {
        std::vector<double> accuracies = classificationAccuracies(options);

        std::cout << "[crossvalidate] accuracies";
        std::for_each(accuracies.begin(), accuracies.end(),
            [](double accuracy) { std::cout << accuracy << " "; });
        std::cout << std::endl;
        double sum = 0;
        std::for_each(accuracies.begin(), accuracies.end(),
            [&sum](double accuracy) { sum += accuracy; });

        if (accuracies.size() != 0) {
            std::cout << "Average accuracy: " << sum / accuracies.size() << std::endl;
        }
    }
    return 0;
}

SOM::SelfOrganizingMap trainSingleSOM(io::argumentOptions options)
{
    SOM::configuration conf = io::SOM::parse_SOM_configuration(options);
    conf.printConfiguration(std::cout);
    digitSet inputSet = io::SOM::parseInputSet(options, conf);

    SOM::SelfOrganizingMap som(inputSet, conf.mapW, conf.mapH);
    if (conf.animation) {
        som.train(conf.maxT, [&](int T, int maxT, SOM::SelfOrganizingMap* map) {
            if ((T + maxT / conf.frameCount) % (maxT / conf.frameCount) == 0) {
                std::ofstream file;
                file.open(
                    conf.animationPath + (std::to_string(T / (maxT / conf.frameCount)) + ".som").c_str());
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
    digitSet& splitTrain, digitSet& splitTest)
{
    digitSet filtered[conf.classCount];
    for (int i = 0; i < conf.classCount; i++) {
        filtered[i] = filterByValue(splitTrain, i);
    }
    std::vector<SOM::SelfOrganizingMap> maps;
    for (int i = 0; i < conf.classCount; i++) {
        maps.push_back(SOM::SelfOrganizingMap(filtered[i], conf.mapW, conf.mapH));
        // maps[i].train(maxT,[&](int T, int maxT, SOM::SelfOrganizingMap *map) {
        // 	if ((T + maxT / frameCount) % (maxT / frameCount) == 0)
        // 	{
        // 		std::ofstream file;
        // 		file.open(animationPath +"/"+
        // std::to_string(i)+"_"+(std::to_string(T / (maxT / frameCount)) +
        // ".som").c_str()); map->printMapToStream(file); 		file.close();
        // 	}
        // });
        maps[i].train(conf.maxT);
    }
    splitTest.minMaxFeatureScale();
    double accuracy = std::count_if(splitTest.getDigits().begin(), splitTest.getDigits().end(),
                          [&](auto& sample) { return SOM::classify(maps, sample); })
        / (float)splitTest.getDigits().size();
    return accuracy;
}

std::vector<double> classificationAccuracies(io::argumentOptions options)
{
    SOM::configuration conf = io::SOM::parse_SOM_configuration(options);
    conf.printConfiguration(std::cout);
    digitSet train = io::SOM::parseInputSet(options, conf);
    digitSet test = io::SOM::parseTestingSet(options, conf);

    std::vector<double> accuracies;
    int k = 5;
    for (int i = 0; i < k; ++i) {
        std::pair<digitSet, digitSet> splitData = crossvalidate_split(train, test, k, i);
        accuracies.push_back(
            sampleClassificationAccuracy(conf, splitData.first, splitData.second));
    }
    return accuracies;
}