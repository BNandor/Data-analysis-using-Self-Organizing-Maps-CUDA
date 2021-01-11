#ifndef OCR_H
#define OCR_H

#include "cuda.cu"
#include "digit.cu"
#include "digitSet.cu"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

/** \brief The namespace of SOM related classes, methods and structures */
/** This namespace contains the SelfOrganizingMap class, a structure that configures the training of it, 
 *  and a method used to classify query points.
 * */
namespace SOM {
    typedef std::vector<std::vector<digit>> SOMContainer;

/** \brief
 * Structure that provides the configuration of the SOM.
 * */

    struct configuration {
        int digitW; /**< Width of the input image in pixels */
        int digitH; /**< Height of the input image in pixels */

        int mapW; /**< Width of the SOM structure. */
        int mapH; /**< Height of the SOM structure. */

        int maxT; /**< Number of maximum iterations during training. */
        bool animation; /**< This flags that the user wants to save intermediate states during training. */
        std::string animationPath; /**< The path to save resulting animation frames to.*/
        int frameCount; /**< The number of frames that need to be saved. */
        bool classification; /**< This flags that classification procedure is enabled. */
        int classCount; /**< The number of classes into which the images have to be categorized in.*/

        /** \brief
         *  Prints all of the configuration parameters into an output stream.
         * */
        void printConfiguration(std::ostream &out) {
            out << "[SOM] image dimensions : " << digitW << " x " << digitH
                << std::endl;
            out << "[SOM] map dimensions : " << mapW << " x " << mapH << std::endl;

            if (animation) {
                out << "[SOM] animation enabled current framecount is " << frameCount
                    << std::endl;
                out << "[SOM] saving animation to  " << animationPath << std::endl;
            }

            if (classification) {
                out << "[SOM] class count is " << classCount << std::endl;
            }
        }
    };

/** \brief 
 * The class that implements the Self Organizing Map model.*/
/** 
 * This class encapsulates a dataset of training points on which 
 * the model is trained and the actual map. The CUDA implementation copies the
 * trainig points and the map to the device memory and operates on them inplace. 
 * After the training finished, the trained map is copied back to the host user memory.
 * */

    class SelfOrganizingMap {
    public:
        /** The constructor takes a set of points and the dimensions of the map as parameter.
         * It normalizes the inputs in order to prevent the explosion of gradients.
         * Afterwards it initializes the map randomly, allocates memory on the CUDA device
         * for the samples and the map and copies them there.
        */
        SelfOrganizingMap(const digitSet &points /**< The training points on which the map will be trained.*/,
                          configuration conf)
                : _map(conf.mapH, std::vector<digit>(conf.mapW)), data(points), mapHeight(conf.mapH),
                  mapWidth(conf.mapW),conf(conf) {
            srand(time(NULL));
            // if (points.getDigits().size() < mapWidth*mapHeight )
            // {
            // 	throw "not enough data to create s.o.m!";
            // }

            featuresMinMax = data.minMaxFeatureScale();
            int digitWidth = conf.digitW;
            int digitHeight = conf.digitW;
            sampleDim = digitWidth * digitHeight;

            // initializeSampledSOM(_map, digitWidth, digitHeight);
            initializeSampledSOM(_map, digitWidth, digitHeight);
            setup_CUDA(sampleDim, data.getDigits().size());
            copy_map_to_device(digitWidth * digitHeight, mapWidth, mapHeight);
            copy_samples_to_device(sampleDim);
        }

        /** The copy constructor of the class copies the samples and map both from the host memory
         * and the CUDA device memory.
         * */
        SelfOrganizingMap(const SelfOrganizingMap &other)
                : data(other.data), _map(other._map) {
            featuresMinMax = other.featuresMinMax;
            mapHeight = other.mapHeight;
            mapWidth = other.mapWidth;
            sampleDim = other.sampleDim;
            conf = other.conf;
            setup_CUDA(sampleDim, data.getDigits().size());
            cudaMemcpy(dev_map, other.dev_map,
                       sizeof(double) * sampleDim * mapHeight * mapWidth,
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_samples, other.dev_samples,
                       sizeof(double) * sampleDim * data.getDigits().size(),
                       cudaMemcpyDeviceToDevice);
        }

        /**
         * Allocates memory on the CUDA device for the provided samples, the map
         * and the intermediary distances
         * */
        void setup_CUDA(int sampleDim /**< The dimensionality of an input sample*/,
                        int sampleCount /**< The number of samples provided to the map*/) {
            cudaMalloc((void **) &dev_samples, sizeof(double) * sampleDim * sampleCount);
            cudaMalloc((void **) &dev_distance, sizeof(double) * mapWidth * mapHeight);
            cudaMalloc((void **) &dev_map,
                       sizeof(double) * sampleDim * mapWidth * mapHeight);
        }

        /** \brief Copies the samples from the host memory to CUDA device memory.*/
        void copy_samples_to_device(int sampleDim) {
            for (int i = 0; i < data.getDigits().size(); ++i) {
                cudaMemcpy((double *) dev_samples + i * sampleDim,
                           data.getDigits()[i].getShades(), sizeof(double) * sampleDim,
                           cudaMemcpyHostToDevice);
            }
        }

        /** \brief Copies the map from the host memory to CUDA device memory.*/
        void copy_map_to_device(int dim, int mapWidth, int mapHeight) {
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    cudaMemcpy((double *) (dev_map + i * mapWidth * dim + j * dim),
                               _map[i][j].getShades(), sizeof(double) * dim,
                               cudaMemcpyHostToDevice);
                }
            }
        }

        /** \brief Copies the resulting trained map from the CUDA device back to the host memory.*/
        void copy_map_from_device(int sampleDim, int mapWidth, int mapHeight) {
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    cudaMemcpy(_map[i][j].getShades(),
                               dev_map + sampleDim * (i * mapWidth + j),
                               sizeof(double) * sampleDim, cudaMemcpyDeviceToHost);
                }
            }
        }

        /** \brief Deallocates all of the allocated memory by the class on the CUDA device. */
        ~SelfOrganizingMap() {
            if (dev_map != nullptr) {
                cudaFree(dev_map);
            }
            if (dev_samples != nullptr) {
                cudaFree(dev_samples);
            }
            if (dev_distance != nullptr) {
                cudaFree(dev_distance);
            }
        }

        /** \brief Initializes the map in a random manner.*/
        /** Initializes each representative node within the map using
         * random values within the minimum and maximum values seen in the sample dataset. */
        void initializeRandomSOM(SOM::SOMContainer &som /**< The map that needs to be initialized.*/,
                                 int dimx /**< The width of the images in pixels.*/,
                                 int dimy /**< The height of the images in pixels.*/) {
            std::for_each(som.begin(), som.end(),
                          [dimx, dimy, this](std::vector<digit> &v) {
                              std::fill(v.begin(), v.end(), digit(dimx, dimy));

                              std::for_each(v.begin(), v.end(), [this](digit &d) {
                                  d.initrandom(featuresMinMax.first, featuresMinMax.second);
                              });
                          });
        }

        /** \brief Initializes the map in a random manner.*/
        /** Initializes each representative node within the map using
         * random samples from the training dataset. */
        void initializeSampledSOM(SOM::SOMContainer &som /**< The map that needs to be initialized.*/,
                                  int dimx /**< The width of the images in pixels.*/,
                                  int dimy /**< The height of the images in pixels.*/) {
            std::for_each(
                    som.begin(), som.end(), [this, dimx, dimy](std::vector<digit> &v) {
                        std::for_each(v.begin(), v.end(), [this](digit &d) {
                            d = (data.getDigits())[rand() % (data.getDigits().size())];
                        });
                    });
        }

        bool safebound(int i, int j) {
            return !(i < 0 || j < 0 || i >= mapHeight || j >= mapWidth);
        }

        /** Return the value of x in the kernel of the Gaussian Probability Density Function. */
        double normal_pdf(double x /**< The value at which the kernel needs to be evaluated.*/,
                          double m /**< The mean of the kernel. */,
                          double s /**< The sigma of the kernel. */) {
            // static const double inv_sqrt_2pi = 0.3989422804014327;
            double a = (x - m) / s;

            // return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
            return std::exp(-0.5f * a * a);
        }

        /** The euclidean distance between two 2D points. */
        double euclideanDistance(int i1, int j1, int i2, int j2) {
            return sqrt((double) (i1 - i2) * (i1 - i2) + (double) (j1 - j2) * (j1 - j2));
        }

        /** Calculates the neighbouring coefficients with a given radius from point (i1,j1) to point (i2,j2). */
        double normalNeighbourCoefficient(int i1 /**<The y index of the first neuron */,
                                          int j1 /**<The x index of the first neuron */,
                                          int i2 /**<The y index of the second neuron */,
                                          int j2 /**<The x index of the second neuron */,
                                          double radius /**<The radius of the neighbouring function. */) {
#ifdef safe

            if (!safebound(i1, j1) || !safebound(i2, j2)) {
                std::cerr << "invalid protoype index" << std::endl;
                exit(1);
            }

#endif
            double euclideanDist = euclideanDistance(i1, j1, i2, j2);

            // return 1.0/normal_pdf(0,0,(std::max(mapWidth,mapHeight)/windowSmallness))
            // * normal_pdf(euclideanDist, 0,
            // (std::max(mapWidth,mapHeight)/windowSmallness));
            return normal_pdf(euclideanDist, 0, radius);
        }

        /** \brief Returns the internal map. */
        SOM::SOMContainer &getMap() { return _map; }

        /** \brief The training method of the map. Results in a trained model on the host memory side. */
        /** This method trains the model for a given number of iterations that it receives as parameter.
         *  It also receives a callback function in order to facilitate the capturing the frames for the animation.
         *  Each iteration it samples a random sample from the dataset and updates the model accordingly.
         *  The best matching unit search is parallelized and so is the neighbourhood update phase.
         *  After the maximum number of iterations it copies the map from CUDA device to host memory.
         * */
        void train(
                int maxT,
                std::function<void(int,  SelfOrganizingMap *)> &&everyFewIterations =
                [](int,  SelfOrganizingMap *) {}) {
            std::cout << "[SOM] starting training" << std::endl;
            double initiallearningrate = 0.9;
            double windowSmallness = 8;
            double neighbourRadius = (std::max(mapWidth, mapHeight) / windowSmallness);
            int T = 0;
            int randomSampleIndex;
            double learningRate;

            std::pair<int, int> closestPrototype;

            while (T < maxT) {
                everyFewIterations(T, this);
                randomSampleIndex = rand() % data.getDigits().size();
                closestPrototype = getClosestPrototypeIndices(randomSampleIndex, sampleDim);
                dev_updateNeighbours<<<mapWidth * mapHeight, 1>>>(
                        closestPrototype.first, closestPrototype.second, learningRate,
                        neighbourRadius, dev_samples + randomSampleIndex * sampleDim,
                        sampleDim, mapWidth, mapHeight, dev_map);

                learningRate = initiallearningrate * normal_pdf(T, 0, maxT / 3);
                neighbourRadius = std::max(0.05, (std::max(mapWidth, mapHeight) / windowSmallness) * (maxT - T) /
                                                 maxT /*normal_pdf( T, 0, maxT)*/);

                T++;
            }
            copy_map_from_device(sampleDim, mapWidth, mapHeight);
            std::cout << "[SOM] Ran for " << T << "generations" << std::endl;
        }

        /** \brief Searches for the sample that is closest to a given neuron within the training data (host side).*/
        digit getClosestSample(int i, int j) {
#ifdef safe
            if (!safebound(i, j)) {
                std::cerr << "invalid protoype index" << std::endl;
                exit(1);
            }
#endif
            double minDist = std::numeric_limits<double>::max();
            double d;
            int minid;
            for (int di = 0; di < data.getDigits().size(); di++) {
                d = _map[i][j] - data.getDigits()[di];
                if (d < minDist) {
                    minDist = d;
                    minid = di;
                }
            }
            return data.getDigits()[minid];
        }

        /** \brief Prints the labeling of the neurons according to the category of their closest sample.*/
        void printMap() {
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    std::cout << getClosestSample(i, j).getValue() << "-";
                }
                std::cout << std::endl;
            }
        }

        /** \brief Prints the map to an output stream.*/
        /** Prints the dimensions of the map, afterwards it prints the actual neuron weights to an output stream.*/
        void printMapToStream(std::ostream &out) {
            // Copy map from device
            out << mapHeight << " " << mapWidth << std::endl;
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    cudaMemcpy(_map[i][j].getShades(),
                               dev_map + sampleDim * (i * mapWidth + j),
                               sizeof(double) * sampleDim, cudaMemcpyDeviceToHost);
                    _map[i][j].appendToFile(out, [](double s) { return s * 255; });
                }
            }
        }

        /** \brief Calculates the distance from a sample to its best matching unit.*/
        double getClosestPrototypeDistance(const digit &sample) {
            double minDist = std::numeric_limits<double>::max();
            double d;

            forEachPrototype([&](digit &proto, int i, int j) {
                d = proto - sample;
                if (d < minDist) {
                    minDist = d;
                }
            });

            return minDist;
        }

        const configuration &getConf() const {
            return conf;
        }

    private:
        void forEachPrototype(std::function<void(digit &, int, int)> &&f) {
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    f(_map[i][j], i, j);
                }
            }
        }

        /** \brief calculates the indices of the neurons that are closest to a random training sample.*/
        std::pair<int, int> getClosestPrototypeIndices(
                int randomSampleIndex /**<The index of the randomly selected sample in the dataset.*/,
                int dim /** the dimensionality of the samples*/) {
            dev_getDistances<<<mapWidth * mapHeight, 1>>>(
                    dev_samples + randomSampleIndex * dim, dev_map, dim, dev_distance);

            double distances[mapWidth * mapHeight];
            cudaMemcpy((double *) distances, dev_distance,
                       sizeof(double) * mapWidth * mapHeight, cudaMemcpyDeviceToHost);

            int maxcuda_i;
            int maxcuda_j;
            double cuminDist = std::numeric_limits<double>::max();

            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    if (*(distances + i * mapWidth + j) < cuminDist) {
                        cuminDist = *(distances + i * mapWidth + j);
                        maxcuda_i = i;
                        maxcuda_j = j;
                    } else {
                        // std::cout<<"dist"<<*(distances + i * mapWidth + j)<<std::endl;
                    }
                }
            }

            if (maxcuda_i >= mapHeight || maxcuda_j >= mapWidth) {
                std::cerr << "Error: please normalize your data properly, could not "
                             "handle distance, it is inf!"
                          << std::endl;
            }

            return std::make_pair(maxcuda_i, maxcuda_j);
        }

        /** Updates the weights of all of the neighbours of the selected best matching unit.*/
        double updateNeighbours(std::pair<int, int> closestPrototype /**< The indices of the best matching unit.*/,
                                double learningRate /** The learning rate of the model.*/,
                                double neighbourRadius /** The radius of the neighbourhood*/,
                                const digit &sample /** The actually selected sample.*/) {
            double closestMaxAbsDistance = sample.minus(_map[closestPrototype.first][closestPrototype.second])
                    .getMaxAbsShade();
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    digit difference = (sample.minus(_map[i][j]));
                    _map[i][j] = _map[i][j] + difference * learningRate *
                                              normalNeighbourCoefficient(closestPrototype.first,
                                                                         closestPrototype.second, i, j,
                                                                         neighbourRadius);
                }
            }
            return closestMaxAbsDistance;
        }

        // Configuration
        SOM::configuration conf;

        digitSet data /**< The set of training datapoints.*/;
        SOM::SOMContainer _map /**< The trained map of the model.*/;
        std::pair<double, double> featuresMinMax /**< The minimum and maximum value over all of the features over all of the datapoints.*/;
        int mapHeight /**< The height of the map.*/;
        int mapWidth /**< The width of the map.*/;
        int sampleDim /**< The dimensionality of each training datapoint.*/;

        // CUDA
        double *dev_map /**<The memory location of the start of the map on the CUDA device.*/;
        double *dev_samples /**<The memory location of the start of the training datapoints on the CUDA device.*/;
        double *dev_distance /**<The memory location of the start of the array of calculated distances on the CUDA device.*/;
    };

/** \brief
 * Classifies a sample image into one of the categories represented by the maps.
 * */
/**
 * This method claculates the distance from the sample point to every map. 
 * The sample is then classified according to the class of the map to which it is closest.
 * */
    int
    classify(std::vector<SelfOrganizingMap> &maps /**< The SOMs that were trained on distinct classes of the dataset.*/,
             const digit &sample /**< The query point that needs to be classified*/) {
        double minDist = std::numeric_limits<double>::max();
        double d;
        int closestMapIndex = -1;
        int i = 0;
        std::for_each(maps.begin(), maps.end(), [&](auto &map) {
            d = map.getClosestPrototypeDistance(sample);

            if (d < minDist) {
                minDist = d;
                closestMapIndex = i;
            }
            i++;
        });
        return closestMapIndex;
    }
} // namespace SOM
#endif