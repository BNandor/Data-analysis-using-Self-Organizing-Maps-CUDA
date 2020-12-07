#include "SOM.cu"
#include "confusion_matrix.cu"
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
    confusion_matrix::matrix confusionMatrix(conf.classCount, std::vector<int>(conf.classCount, 0));
    std::for_each(splitTest.getDigits().begin(), splitTest.getDigits().end(),
        [&](auto& sample) { ++confusionMatrix[SOM::classify(maps, sample)][sample.getValue()]; });
    confusion_matrix::print_matrix(std::cout, confusionMatrix, conf.classCount);
    double accuracy = confusion_matrix::accuracy(confusionMatrix, conf.classCount);

    // Precision
    std::vector<double> positive_precisions = confusion_matrix::positive_precisions(confusionMatrix,conf.classCount);
    std::cout<<"Precisions"<<std::endl;
    std::for_each(positive_precisions.begin(),positive_precisions.end(),[](double pr){std::cout<<" "<<pr;});
    std::cout<<std::endl;

    // Sensitivity
    std::vector<double> sensitivities = confusion_matrix::sensitivity(confusionMatrix,conf.classCount);
    std::cout<<"Sensitivities"<<std::endl;
    std::for_each(sensitivities.begin(),sensitivities.end(),[](double se){std::cout<<" "<<se;});
    std::cout<<std::endl;

     // Fscore
     std::vector<double> fscores = confusion_matrix::fscore(positive_precisions,sensitivities);
     std::cout<<"Fscores"<<std::endl;
     std::for_each(fscores.begin(),fscores.end(),[](double fs){std::cout<<" "<<fs;});
     std::cout<<std::endl;
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