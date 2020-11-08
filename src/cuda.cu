#ifndef CUDASOM_H
#define CUDASOM_H

#include <math.h>

__global__ void kernel(double* sample,double* map,int dim, double* distance){
	int offset = blockIdx.x;
	double* protoype = map + offset*dim;
	double sum = 0;
	for(int c=0;c<dim;c++){
		sum+=(sample[c]-protoype[c])* (sample[c]-protoype[c]);
	}
	distance[offset] = sqrt(sum);
}

#endif