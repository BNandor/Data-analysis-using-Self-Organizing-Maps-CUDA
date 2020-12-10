#ifndef IO_SOM_H
#define IO_SOM_H
#include "SOM.cu"
#include <iostream>
#include <map>

#define DEFAULT_TRAINING_FILENAME "optdigits.tra"
#define DEFAULT_TEST_FILENAME "optdigits.tes"

#ifndef OUTPUT_SOM_FILENAME
#define OUTPUT_SOM_FILENAME "som.txt"
#endif

#ifndef DEFAULT_MAP_WIDTH
#define DEFAULT_MAP_WIDTH 10
#endif

#ifndef DEFAULT_MAP_HEIGHT
#define DEFAULT_MAP_HEIGHT 10
#endif

#ifndef ANIMATION_DEFAULT_FRAMECOUNT
#define ANIMATION_DEFAULT_FRAMECOUNT 100
#endif

#ifndef CLASSIFICATION_DEFAULT_CLASS_COUNT
#define CLASSIFICATION_DEFAULT_CLASS_COUNT 10
#endif

/** This namespace holds methods needed for the parsing of argument options and reading
 *  of training and testing datasets.*/ 
namespace io {

/** \brief This type hides the container type of the argument options.*/
typedef std::map<std::string, std::string> argumentOptions;

/** Returns an argument options container that contains the configuration options 
 * passed to the executable via CLI.*/
argumentOptions parse_options(int argc, const char* argv[])
{
    argumentOptions options;
    for (int i = 1; i < argc; i++) {
        int eq = std::string(argv[i]).find("=");
        if (eq != std::string::npos) {
            std::string option = std::string(argv[i]).substr(0, eq);
            if (strlen(argv[i]) > eq + 1) {
                std::string value = std::string(argv[i]).substr(eq + 1, strlen(argv[i]));
                options[option] = value;
            }
        }
    }
    return options;
}

/** This sub-namespace contains the parsing of options needed for the configuring of the SOM.*/
namespace SOM {

    /** Returns a ::SOM::configuration read from CLI.*/
    ::SOM::configuration parse_SOM_configuration(argumentOptions options)
    {
        ::SOM::configuration conf;
        conf.digitW = options.count("imagew") > 0 ? std::atoi(options["imagew"].c_str()) : 8;
        conf.digitH = options.count("imageh") > 0 ? std::atoi(options["imageh"].c_str()) : 8;

        conf.mapW = options.count("mapw") > 0 ? std::atoi(options["mapw"].c_str())
                                              : DEFAULT_MAP_WIDTH;
        conf.mapH = options.count("maph") > 0 ? std::atoi(options["maph"].c_str())
                                              : DEFAULT_MAP_HEIGHT;

        conf.maxT = options.count("gen") > 0 ? std::atoi(options["gen"].c_str()) : 3000;

        conf.animation = options.count("animation") > 0 || options.count("animationPath") > 0;
        conf.animationPath = options.count("animationPath") > 0 ? options["animationPath"] : "./";
        conf.frameCount = options.count("framecount") > 0
            ? std::atoi(options["framecount"].c_str())
            : ANIMATION_DEFAULT_FRAMECOUNT;

        conf.classification = options.count("test") > 0;
        conf.classCount = options.count("classCount") > 0
            ? std::atoi(options["classCount"].c_str())
            : CLASSIFICATION_DEFAULT_CLASS_COUNT;
        return conf;
    }

    /** Returns a training dataset read from files provided by the CLI arguments and SOM configuration.*/
    digitSet parseInputSet(argumentOptions options, ::SOM::configuration conf)
    {
        std::string inputName = DEFAULT_TRAINING_FILENAME;

        if (!options.count("input")) {
            std::cerr << "[SOM] please specify an input file with  input=file"
                      << std::endl;
            exit(0);
        } else {
            inputName = options["input"];
        }

        std::fstream input;
        input.open(inputName.c_str());

        std::cout << "[SOM] reading input from " << inputName << std::endl;

        if (!input.is_open()) {
            std::cerr << "could not open file" << inputName << std::endl;
            exit(-1);
        }

        digitSet inputSet(conf.digitW, conf.digitH);
        input >> inputSet;
        return inputSet;
    }

    /** Returns a testing dataset read from files provided by the CLI arguments and SOM configuration.*/
    digitSet parseTestingSet(argumentOptions options, ::SOM::configuration conf)
    {
        std::string testName = options["test"];

        std::fstream testinput;
        testinput.open(testName);

        std::cout << "[SOM] reading test data from " << testName << std::endl;
        if (!testinput.is_open()) {
            std::cerr << "could not open file" << testName << std::endl;
            exit(-1);
        }
        digitSet test(conf.digitW, conf.digitH);
        testinput >> test;
        return test;
    }
} // namespace SOM
} // namespace io
#endif