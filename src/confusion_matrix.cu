#ifndef CONFUSION_MATRIX_PROCESSING_H
#define CONFUSION_MATRIX_PROCESSING_H
#include <algorithm>
#include <iostream>
#include <utility>
/**
 * 
 * 
 * 
 */

namespace confusion_matrix {
typedef std::vector<std::vector<int>> matrix;

void print_matrix(std::ostream& out, matrix m, int dim)
{
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

double accuracy(matrix m, int dim)
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

std::vector<double> positive_precisions(matrix m, int dim)
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

std::vector<double> negative_precisions(matrix m, int dim)
{
    std::vector<double> precisions;
    // int ALL = 0;
    // for (int i = 0; i < dim; i++) {
    //     for (int j = 0; j < dim; j++) {
    //         ALL += m[i][j];
    //     }
    // }

    // for (int i = 0; i < dim; i++) {
    //     int TP = m[i][i];
    //     int FP = 0;
    //     for (int j = 0; j < dim; j++) {
    //         FP += m[i][j];
    //     }
    //     FP -= TP;
    //     int TN = ALL - TP - FP;

    //     precisions.push_back(TP / (double)(TP + FP));
    // }
    return precisions;
}

std::vector<double> sensitivity(matrix m, int dim)
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

std::vector<double> specificity(matrix m, int dim)
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

std::vector<double> fscore(std::vector<double> precisions, std::vector<double> sensitivities)
{
    std::vector<double> fscores;
    if (precisions.size() != sensitivities.size()) {
        std::cerr << "[confusion_matrix::fscore] invalida precisions or sensitivities" << std::endl;
        exit(1);
    }

    for (int i = 0; i < sensitivities.size(); i++) {
        fscores.push_back(2 / ((1.0 / precisions[i]) + (1.0 / sensitivities[i])));
    }
    return fscores;
}

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

std::pair<double, double> confidence95(std::vector<double> samples)
{
    if (samples.size() == 1) {
        return std::make_pair(0, 0);
    }
    double sum = 0;
    std::for_each(samples.begin(), samples.end(),
        [&sum](double sample) { sum += sample; });
    double mean = sum / samples.size();
    double sampleStandardDeviation = sqrt(sampleVariance(samples, mean));
    double standardError = sampleStandardDeviation / sqrt((double)samples.size());
    return std::make_pair(mean, 1.96 * standardError);
}

std::vector<double> AUC(matrix m, int dim)
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