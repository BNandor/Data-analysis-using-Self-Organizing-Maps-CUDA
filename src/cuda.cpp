#ifndef CUDASOM_H
#define CUDASOM_H

#include <math.h>

__device__ double distance(double* a, double* b, int dim)
{
    double sum = 0;
    double d;
    for (int c = 0; c < dim; c++) {
        d = (a[c] - b[c]);
        sum += d * d;
    }
    return sqrt(sum);
}

__device__ double* difference(double* a, double* b, int dim)
{
    double* diff = (double*)malloc(sizeof(double) * dim);
    for (int c = 0; c < dim; c++) {
        diff[c] = (a[c] - b[c]);
    }
    return diff;
}

__global__ void dev_getDistances(double* sample, double* map, int dim,
    double* dist)
{
    int offset = blockIdx.x;
    double* protoype = map + offset * dim;
    dist[offset] = distance(sample, protoype, dim);
}

__device__ double normal_pdf(double x, double m, double s)
{
    double a = (x - m) / s;
    return exp(-0.5f * a * a);
}

__device__ double euclideanDistance(int i1, int j1, int i2, int j2)
{
    return sqrt((double)(i1 - i2) * (i1 - i2) + (double)(j1 - j2) * (j1 - j2));
}

__device__ double normalNeighbourCoefficient(int i1, int j1, int i2, int j2,
    double radius)
{
    double euclideanDist = euclideanDistance(i1, j1, i2, j2);
    return normal_pdf(euclideanDist, 0, radius);
}

__global__ void dev_updateNeighbours(int closest_i, int closest_j,
    double learningRate,
    double neighbourRadius, double* sample,
    int dim, int mapWidth, int mapHeight,
    double* map)
{
    int offset = blockIdx.x;
    double* protoype = map + offset * dim;
    double* diff = difference(sample, protoype, dim);
    double ncof = normalNeighbourCoefficient(closest_i, closest_j, offset / mapWidth,
        offset % mapWidth, neighbourRadius);
    for (int c = 0; c < dim; c++) {
        protoype[c] = protoype[c] + diff[c] * learningRate * ncof;
    }
    free(diff);
}

#endif