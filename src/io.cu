#ifndef IO_SOM_H
#define IO_SOM_H
#define OPTION_FLAG_COUNT_CLASSIFICATION_LABEL "classCount"

#include "SOM.cu"
#include <iostream>
#include <map>
#include "constants.cuh"

/** This namespace holds methods needed for the parsing of argument options and reading
 *  of training and testing datasets.*/
namespace io {

/** \brief This type hides the container type of the argument options.*/
    typedef std::map<std::string, std::string> argumentOptions;

/** Returns an argument options container that contains the configuration options 
 * passed to the executable via CLI.*/
    argumentOptions parse_options_cli(int argc, const char **argv) {
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

/** Returns an argument options container that contains the configuration options
 * passed to the executable via configuration file.*/
    argumentOptions parse_options_configfile(std::ifstream &in) {
        argumentOptions options;
        std::string line;
        while (getline(in, line)) {
            size_t start = line.find_first_not_of(" \n\r\t\f\v");
            size_t end = line.find_last_not_of(" \n\r\t\f\v");
            line = (end == std::string::npos) ? "" : line.substr(0,end+1);
            line = (start == std::string::npos) ? "" : line.substr(start);
            int eq = line.find('=');
            if (eq != std::string::npos) {
                std::string option = line.substr(0, eq);
                if (line.size() > eq + 1) {
                    std::string value = line.substr(eq + 1, line.size());
                    options[option] = value;
                }
            }
        }
        return options;
    }
/** This sub-namespace contains the parsing of options needed for the configuring of the SOM.*/
    namespace SOM {

        /** Returns a ::SOM::configuration read from CLI.*/
        ::SOM::configuration parse_SOM_configuration(argumentOptions options) {
            ::SOM::configuration conf;
            conf.digitW = options.count(OPTION_FLAG_DIM_WIDTH_IMAGE) > 0 ? std::atoi(
                    options[OPTION_FLAG_DIM_WIDTH_IMAGE].c_str()) : 8;
            conf.digitH = options.count(OPTION_FLAG_DIM_HEIGHT_IMAGE) > 0 ? std::atoi(
                    options[OPTION_FLAG_DIM_HEIGHT_IMAGE].c_str()) : 8;

            conf.mapW =
                    options.count(OPTION_FLAG_DIM_WIDTH_MAP) > 0 ? std::atoi(options[OPTION_FLAG_DIM_WIDTH_MAP].c_str())
                                                                 : DEFAULT_MAP_WIDTH;
            conf.mapH = options.count(OPTION_FLAG_DIM_HEIGHT_MAP) > 0 ? std::atoi(
                    options[OPTION_FLAG_DIM_HEIGHT_MAP].c_str())
                                                                      : DEFAULT_MAP_HEIGHT;

            conf.maxT = options.count(OPTION_FLAG_COUNT_TRAIN_SOM_MAX_ITERATION) > 0 ? std::atoi(
                    options[OPTION_FLAG_COUNT_TRAIN_SOM_MAX_ITERATION].c_str()) : 3000;

            conf.animation =
                    options.count(OPTION_FLAG_ON_ANIMATION) > 0 || options.count(OPTION_FLAG_PATH_OUTPUT_ANIMATION) > 0;
            conf.animationPath =
                    options.count(OPTION_FLAG_PATH_OUTPUT_ANIMATION) > 0 ? options[OPTION_FLAG_PATH_OUTPUT_ANIMATION]
                                                                         : "./";
            conf.frameCount = options.count(OPTION_FLAG_COUNT_ANIMATION_FRAMES) > 0
                              ? std::atoi(options[OPTION_FLAG_COUNT_ANIMATION_FRAMES].c_str())
                              : ANIMATION_DEFAULT_FRAMECOUNT;

            conf.classification = options.count(OPTION_FLAG_PATH_TEST_INPUT) > 0;
            conf.classCount = options.count(OPTION_FLAG_COUNT_CLASSIFICATION_LABEL) > 0
                              ? std::atoi(options[OPTION_FLAG_COUNT_CLASSIFICATION_LABEL].c_str())
                              : CLASSIFICATION_DEFAULT_CLASS_COUNT;
            return conf;
        }

        /** Returns a training dataset read from files provided by the CLI arguments and SOM configuration.*/
        digitSet parseInputSet(argumentOptions options, ::SOM::configuration conf) {
            std::string inputName = DEFAULT_TRAINING_FILENAME;

            if (!options.count(OPTION_FLAG_PATH_TRAIN_INPUT)) {
                std::cerr << "[SOM] please specify an input file with  input=file"
                          << std::endl;
                exit(0);
            } else {
                inputName = options[OPTION_FLAG_PATH_TRAIN_INPUT];
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
        digitSet parseTestingSet(argumentOptions options, ::SOM::configuration conf) {
            std::string testName = options[OPTION_FLAG_PATH_TEST_INPUT];

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

        /** Writes a SOM to a file specified at outputName*/
        void writeSOM(std::string outputName, ::SOM::SelfOrganizingMap &som) {
            std::ofstream output;
            output.open(outputName);
            som.printMapToStream(output);
            output.close();
        }

        /** Prints intermediate SOM */
        void printIntermediateSOM(int T, ::SOM::SelfOrganizingMap* map,std::string output) {
                std::ofstream file;
                file.open(output);
                map->printMapToStream(file);
                file.close();
        }
    } // namespace SOM
} // namespace io
#endif