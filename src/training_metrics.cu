#ifndef CONFUSION_MATRIX_PROCESSING_H
#define CONFUSION_MATRIX_PROCESSING_H
#include <algorithm>
#include <iostream>
#include <utility>

/** \brief
 * This namespace contains the methods that derive certain metrics 
 * from a provided confusion confusion_matrix.
 */
namespace training_metrics {

/** \brief This type hides the confusion confusion_matrix container implementation type. */
typedef std::vector<std::vector<int>> confusion_matrix;

/** \brief Prints the confusion confusion_matrix to an output stream.
 * */
void print_confusion_matrix(std::ostream& out, confusion_matrix m, int dim)
{
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

/** \brief calculates the overall accuracy of the classifier. */
double accuracy(confusion_matrix m, int dim)
{
    int TP = 0;
    for (int i = 0; i < dim; i++) {
        TP += m[i][i];
    }
    int ALL = 0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            ALL += m[i][j];
        }
    }
    return TP / (double)ALL;
}

/**  Calculates the positive precision (TP / (TP + FN)) of the classifier.
 * Employs a one versus all strategy, resulting in a value for every class.
 */
std::vector<double> positive_precisions(confusion_matrix m, int dim)
{
    std::vector<double> precisions;
    for (int i = 0; i < dim; i++) {
        int TP = m[i][i];
        int FN = 0;
        for (int j = 0; j < dim; j++) {
            FN += m[i][j];
        }
        FN -= TP;

        precisions.push_back(TP / (double)(TP + FN));
    }
    return precisions;
}

/**  Calculates the positive sensitivity (TP / (TP + FP)) of the classifier.
 * Employs a one versus all strategy, resulting in a value for every class.
 */
std::vector<double> sensitivity(confusion_matrix m, int dim)
{
    std::vector<double> sensitivities;
    for (int i = 0; i < dim; i++) {
        int TP = m[i][i];
        int FP = 0;
        for (int j = 0; j < dim; j++) {
            FP += m[j][i];
        }
        FP -= TP;

        sensitivities.push_back(TP / (double)(TP + FP));
    }
    return sensitivities;
}

/**  Calculates the specificity (TN / (TN + FP)) of the classifier. 
 * Employs a one versus all strategy, resulting in a value for every class.
 * */
std::vector<double> specificity(confusion_matrix m, int dim)
{
    std::vector<double> specificities;
    int ALL = 0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            ALL += m[i][j];
        }
    }

    for (int i = 0; i < dim; i++) {
        int TP = m[i][i];
        int FP = 0;
        for (int j = 0; j < dim; j++) {
            FP += m[j][i];
        }
        FP -= TP;
        int FN = 0;
        for (int j = 0; j < dim; j++) {
            FN += m[i][j];
        }
        FN -= TP;
        int TN = ALL - FP - FN - TP;

        specificities.push_back(TN / (double)(TN + FP));
    }
    return specificities;
}

/**  Calculates the fscore (2 / ((1/precision) + (1/sensitivity)) )) of the classifier. 
* Employs a one versus all strategy, resulting in a value for every class.
*/
std::vector<double> fscore(std::vector<double> precisions /**< The precision of every class. */,
    std::vector<double> sensitivities /**< The sensitivity of every class. */)
{
    std::vector<double> fscores;
    if (precisions.size() != sensitivities.size()) {
        std::cerr << "[training_metrics::fscore] invalida precisions or sensitivities" << std::endl;
        exit(1);
    }

    for (int i = 0; i < sensitivities.size(); i++) {
        fscores.push_back(2 / ((1.0 / precisions[i]) + (1.0 / sensitivities[i])));
    }
    return fscores;
}

/**  Clalculates the variance of some samples, with the mean already computed.*/
double sampleVariance(std::vector<double> samples, double mean)
{
    if (samples.size() == 1) {
        return 0;
    }
    double sum = 0;
    std::for_each(samples.begin(), samples.end(),
        [&sum, &mean](double sample) { sum += (sample - mean) * (sample - mean); });
    return sum / (samples.size() - 1);
}

/**  Clalculates the variance of some samples.*/
double sampleVariance(std::vector<double> samples)
{
    if (samples.size() == 1) {
        return 0;
    }
    double sum = 0;
    std::for_each(samples.begin(), samples.end(),
        [&sum](double sample) { sum += sample; });
    double mean = sum / samples.size();
    sum = 0;
    std::for_each(samples.begin(), samples.end(),
        [&sum, &mean](double sample) { sum += (sample - mean) * (sample - mean); });
    return sum / (samples.size() - 1);
}

/** Returns a 95% confidence interval to the mean of a particular set of samples.
 *  A tuple (mu, error) is returned with mu being the mean and error being the +-error.
*/
std::pair<double, double> confidence95(std::vector<double> samples)
{
    if (samples.size() == 1) {
        return std::make_pair(samples[0], 0);
    }
    double sum = 0;
    std::for_each(samples.begin(), samples.end(),
        [&sum](double sample) { sum += sample; });
    double mean = sum / samples.size();
    double sampleStandardDeviation = sqrt(sampleVariance(samples, mean));
    double standardError = sampleStandardDeviation / sqrt((double)samples.size());
    return std::make_pair(mean, 1.96 * standardError);
}

/** Calculates the Area Under Curve of each class for a particular classifier.*/
std::vector<double> AUC(confusion_matrix m, int dim)
{
    std::vector<double> sensitivities = sensitivity(m, dim);
    std::vector<double> specificities = specificity(m, dim);
    std::vector<double> areas;

    for (int i = 0; i < dim; i++) {
        areas.push_back((sensitivities[i] + specificities[i]) / 2);
    }
    return areas;
}

}
#endif