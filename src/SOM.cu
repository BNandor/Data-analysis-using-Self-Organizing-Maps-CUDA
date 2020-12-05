#ifndef OCR_H
#define OCR_H
#include <fstream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits>
#include <sstream>
#include <functional>
#include <iostream>
#include "digit.cu"
#include "digitSet.cu"
#include "cuda.cu"

namespace SOM {
	class SelfOrganizingMap
	{
	public:
		SelfOrganizingMap(const digitSet &points, int mapWidth, int mapHeight) : _map(mapHeight, std::vector<digit>(mapWidth)), data(points), mapHeight(mapHeight), mapWidth(mapWidth)
		{
			srand(time(NULL));
			// if (points.getDigits().size() < mapWidth*mapHeight )
			// {
			// 	throw "not enough data to create s.o.m!";
			// }

			featuresMinMax = data.minMaxFeatureScale();
			int digitWidth = points.getDigit(0).getWidth();
			int digitHeight = points.getDigit(0).getHeight();
			sampleDim = digitWidth*digitHeight;

			// initializeSampledSOM(_map, digitWidth, digitHeight);
			initializeRandomSOM(_map, digitWidth, digitHeight);
			setup_CUDA(sampleDim,data.getDigits().size());
			copy_map_to_device(digitWidth*digitHeight,mapWidth,mapHeight);
			copy_samples_to_device(sampleDim);
		}

		SelfOrganizingMap(const SelfOrganizingMap& other) {
			data = other.data;
			_map = other._map;

			featuresMinMax = other.featuresMinMax;
			mapHeight = other.mapHeight;
			mapWidth = other.mapWidth;
			sampleDim = other.sampleDim;
			setup_CUDA(sampleDim,data.getDigits().size());
			cudaMemcpy(dev_map,other.dev_map,sizeof(double)*sampleDim*mapHeight*mapWidth,cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_samples,other.dev_samples,sizeof(double)*sampleDim*data.getDigits().size(),cudaMemcpyDeviceToDevice);
		}

		void setup_CUDA(int sampleDim,int sampleCount) {
			cudaMalloc((void**)&dev_samples,sizeof(double)*sampleDim*sampleCount);
			cudaMalloc((void**)&dev_distance,sizeof(double)*mapWidth*mapHeight);
			cudaMalloc((void**)&dev_map,sizeof(double)*sampleDim* mapWidth * mapHeight);
		}

		void copy_samples_to_device (int sampleDim){
			for(int i=0;i<data.getDigits().size();++i) {
				cudaMemcpy((double*)dev_samples + i*sampleDim, data.getDigits()[i].getShades(),sizeof(double)*sampleDim,cudaMemcpyHostToDevice);
			}
		}

		void copy_map_to_device(int dim, int mapWidth, int mapHeight){
			for(int i=0;i<mapHeight;i++){
				for(int j=0;j<mapWidth;j++){
					cudaMemcpy((double*)(dev_map + i*mapWidth*dim + j*dim),_map[i][j].getShades(),sizeof(double)*dim,cudaMemcpyHostToDevice);
				}
			}
		}

		void copy_map_from_device (int sampleDim, int mapWidth, int mapHeight ){
			for (int i = 0; i < mapHeight; i++)
			{
				for (int j = 0; j < mapWidth; j++)
				{
					cudaMemcpy(_map[i][j].getShades(),dev_map + sampleDim*(i*mapWidth + j),sizeof(double)*sampleDim,cudaMemcpyDeviceToHost);
				}
			}
		}

		~SelfOrganizingMap() {
			if(dev_map != nullptr){
				cudaFree(dev_map);
			}
			if(dev_samples != nullptr ) {
				cudaFree(dev_samples);
			}
			if(dev_distance != nullptr ) {
				cudaFree(dev_distance);
			}
		}

		void initializeRandomSOM(std::vector<std::vector<digit>> &som, int dimx, int dimy)
		{
			std::for_each(som.begin(), som.end(), [dimx, dimy, this](std::vector<digit> &v) {
				std::fill(v.begin(), v.end(), digit(dimx, dimy));

				std::for_each(v.begin(), v.end(), [this](digit &d) {
					d.initrandom(featuresMinMax.first,featuresMinMax.second);
				});
			});
		}

		void initializeSampledSOM(std::vector<std::vector<digit>> &som, int dimx, int dimy)
		{
			std::for_each(som.begin(), som.end(), [this, dimx, dimy](std::vector<digit> &v) {
				std::for_each(v.begin(), v.end(), [this](digit &d) {
					d = (data.getDigits())[rand() % (data.getDigits().size())];
				});
			});
		}

		bool safebound(int i, int j)
		{
			return !(i < 0 || j < 0 || i >= mapHeight || j >= mapWidth);
		}

		double normal_pdf(double x, double m, double s)
		{
			// static const double inv_sqrt_2pi = 0.3989422804014327;
			double a = (x - m) / s;

			// return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
			return std::exp(-0.5f * a * a);
		}

		double euclideanDistance(int i1, int j1, int i2, int j2)
		{
			return sqrt((double)(i1 - i2) * (i1 - i2) + (double)(j1 - j2) * (j1 - j2));
		}

		double normalNeighbourCoefficient(int i1, int j1, int i2, int j2, double radius)
		{
	#ifdef safe

			if (!safebound(i1, j1) || !safebound(i2, j2))
			{
				std::cerr << "invalid protoype index" << std::endl;
				exit(1);
			}

	#endif
			double euclideanDist = euclideanDistance(i1, j1, i2, j2);

			// return 1.0/normal_pdf(0,0,(std::max(mapWidth,mapHeight)/windowSmallness)) * normal_pdf(euclideanDist, 0, (std::max(mapWidth,mapHeight)/windowSmallness));
			return normal_pdf(euclideanDist, 0, radius);
		}

		std::vector<std::vector<digit>> &getMap() { return _map; }

		void train(int maxT,std::function<void(int,int,SelfOrganizingMap*)> &&everyFewIterations=[](int,int,SelfOrganizingMap*){})
		{
			std::cout << "[SOM] starting training" << std::endl;
			double initiallearningrate = 0.9;
			double windowSmallness = 8;
			double neighbourRadius = (std::max(mapWidth, mapHeight) / windowSmallness);
			int T = 0;
			double minAdjustment = 0.1;
			double maxAdjusted = minAdjustment + 1;
			int randomSampleIndex;
			double learningRate;

			std::pair<int, int> closestPrototype;

			while (T < maxT /*&& maxAdjusted > minAdjustment*/)
			{
				everyFewIterations(T,maxT,this);
				randomSampleIndex = rand() % data.getDigits().size();
				closestPrototype = getClosestPrototypeIndices(randomSampleIndex,sampleDim);
				
				// cudaMemcpy(dev_sample,data.getDigits()[randomSampleIndex].getShades(),sizeof(double)*sampleDim,cudaMemcpyHostToDevice);
				dev_updateNeighbours<<<mapWidth*mapHeight,1>>>(closestPrototype.first, closestPrototype.second, learningRate, neighbourRadius, dev_samples+randomSampleIndex*sampleDim, sampleDim,  mapWidth, mapHeight, dev_map);

				learningRate = initiallearningrate * normal_pdf(T, 0, maxT / 3);
				neighbourRadius = std::max(0.05, (std::max(mapWidth, mapHeight) / windowSmallness) * (maxT - T) / maxT /*normal_pdf( T, 0, maxT)*/);

				T++;
			}
			copy_map_from_device(sampleDim,mapWidth,mapHeight);
			std::cout << "[SOM] Ran for " << T << "generations" << std::endl;
		}

		digit getClosestSample(int i, int j)
		{
	#ifdef safe
			if (!safebound(i, j))
			{
				std::cerr << "invalid protoype index" << std::endl;
				exit(1);
			}
	#endif
			double minDist = std::numeric_limits<double>::max();
			double d;
			int minid;
			for (int di = 0; di < data.getDigits().size(); di++)
			{
				d = _map[i][j] - data.getDigits()[di];
				if (d < minDist)
				{
					minDist = d;
					minid = di;
				}
			}
			return data.getDigits()[minid];
		}

		void printMap()
		{
			// Copy map from device
			for (int i = 0; i < mapHeight; i++)
			{
				for (int j = 0; j < mapWidth; j++)
				{
					std::cout << getClosestSample(i, j).getValue() << "-";
				}
				std::cout << std::endl;
			}
		}

		void printMapToStream(std::ostream &out)
		{
			// Copy map from device
			out << mapHeight << " " << mapWidth << std::endl;
			for (int i = 0; i < mapHeight; i++)
			{
				for (int j = 0; j < mapWidth; j++)
				{
					cudaMemcpy(_map[i][j].getShades(),dev_map + sampleDim*(i*mapWidth + j),sizeof(double)*sampleDim,cudaMemcpyDeviceToHost);
					_map[i][j].appendToFile(out,[](double s){return s*255;});
				}
			}
		}

		double getClosestPrototypeDistance(const digit &sample)
		{
			double minDist = std::numeric_limits<double>::max();
			double d;
			
			forEachPrototype([&](digit &proto, int i, int j) {
				d = proto - sample;
				if (d < minDist)
				{
					minDist = d;
				}
			});

			return minDist;
		}


	private:

		void forEachPrototype(std::function<void(digit &, int, int)> &&f)
		{
			for (int i = 0; i < mapHeight; i++)
			{
				for (int j = 0; j < mapWidth; j++)
				{
					f(_map[i][j], i, j);
				}
			}
		}
		
		std::pair<int, int> getClosestPrototypeIndices(int randomSampleIndex, int dim)
		{
				dev_getDistances<<<mapWidth*mapHeight,1>>>(dev_samples+randomSampleIndex*dim,dev_map,dim,dev_distance);

				double distances[mapWidth*mapHeight];
				cudaMemcpy((double*)distances,dev_distance,sizeof(double)*mapWidth*mapHeight,cudaMemcpyDeviceToHost);
				
				int maxcuda_i;
				int maxcuda_j;
				double cuminDist = std::numeric_limits<double>::max();

				for(int i=0;i<mapHeight;i++){
					for(int j=0;j<mapWidth;j++){
						if(*(distances + i*mapWidth +j) < cuminDist){
							cuminDist =*(distances + i*mapWidth +j);
							maxcuda_i = i;
							maxcuda_j = j;
						}
					}
				}

				if(maxcuda_i >= mapHeight || maxcuda_j >= mapWidth){
					std::cerr<<"Error: please normalize your data properly, could not handle distance, it is inf!"<<std::endl;
				}

			return std::make_pair(maxcuda_i, maxcuda_j);
		}

		double updateNeighbours(std::pair<int, int> closestPrototype, double learningRate, double neighbourRadius, const digit &sample)
		{
			double closestMaxAbsDistance = sample.minus(_map[closestPrototype.first][closestPrototype.second]).getMaxAbsShade();
			for (int i = 0; i < mapHeight; i++)
			{
				for (int j = 0; j < mapWidth; j++)
				{
					digit difference = (sample.minus(_map[i][j]));
					_map[i][j] = _map[i][j] + difference * learningRate * normalNeighbourCoefficient(closestPrototype.first, closestPrototype.second, i, j, neighbourRadius);
				}
			}
			return closestMaxAbsDistance;
		}

		digitSet data;
		std::vector<std::vector<digit>> _map;
		std::pair<double,double> featuresMinMax;
		int mapHeight;
		int mapWidth;
		int sampleDim;
		// CUDA
		double* dev_map;
		double* dev_samples;
		double* dev_distance;
	};

	struct configuration {
		
		int digitW;
		int digitH;
	
		int mapW;
		int mapH;
	
		int maxT;
		bool animation;
		std::string animationPath;
		int frameCount;
		bool classification;
		int classCount;

		void printConfiguration(std::ostream& out) {
			out << "[SOM] image dimensions : " << digitW << " x " << digitH << std::endl;
			out << "[SOM] map dimensions : " << mapW << " x " << mapH << std::endl;

			if(animation){
				out<<"[SOM] animation enabled current framecount is "<<frameCount<<std::endl;
				out<<"[SOM] saving animation to  "<<animationPath<<std::endl;
			}

			if(classification) {
				out << "[SOM] class count is " << classCount << std::endl;
			}

		}	
	}; 

	bool classify(std::vector<SelfOrganizingMap> &maps, const digit &sample)
	{
		double minDist = std::numeric_limits<double>::max();
		double d;
		int closestMapIndex = -1;
		int i = 0;
		std::for_each(maps.begin(), maps.end(), [&](auto &map) {
			d = map.getClosestPrototypeDistance(sample);

			if (d < minDist)
			{
				minDist = d;
				closestMapIndex = i;
			}
			i++;
		});
		return sample.getValue() == closestMapIndex;
	}
}
#endif