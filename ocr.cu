#ifndef OCR_H
#define OCR_H
#include <fstream>
#include <vector>
#include <string.h>
#include <armadillo>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits>

class digit
{
private:
	int value;
	double *shades;
	int _width;
	int _height;

public:
	int dimension() const { return _width * _height; }
	digit(int width = 8, int height = 8);
	~digit();
	digit(const digit &other)
	{
		shades = new double[other._width * other._height];
		memcpy(shades, other.shades, sizeof(double) * (other._width * other._height));
		_width = other._width;
		_height = other._height;
		value = other.value;
	}

	void operator=(const digit &other)
	{
		if (shades)
			delete shades;
		shades = new double[other._width * other._height];
		memcpy(shades, other.shades, sizeof(double) * (other._width * other._height));
		_width = other._width;
		_height = other._height;
		value = other.value;
	}

	int getWidth() const { return _width; }
	int getHeight() const { return _height; }
	double * getShades()const{return shades;}
	void initrandom()
	{
		for (int i = 0; i < dimension(); i++)
		{
			shades[i] = rand() % 256;
		}
	}

	std::ostream &appendToFile(std::ostream &out)
	{
		for (int i = 0; i < _width * _height; i++)
		{
			out << (int)(shades[i]*255) << " ";
		}
		out << std::endl;
		return out;
	}

	digit operator+(const digit &other) const
	{
#ifdef safe
		if (_width != other._width || _height != other._height)
		{
			std::cerr << "dimension discrepancy" << std::endl;
			return -1;
		}
#endif
		digit sum(*this);
		for (int i = 0; i < _width * _height; i++)
		{
			sum.shades[i] += other.shades[i];
		}
		return sum;
	}
	digit minus(const digit &other) const
	{
#ifdef safe
		if (_width != other._width || _height != other._height)
		{
			std::cerr << "dimension discrepancy" << std::endl;
			return -1;
		}
#endif
		digit difference(*this);
		for (int i = 0; i < _width * _height; i++)
		{
			difference.shades[i] -= other.shades[i];
		}
		return difference;
	}

	digit operator*(double scalar) const
	{
		digit d(*this);
		for (int i = 0; i < _width * _height; i++)
		{
			d.shades[i] *= scalar;
		}
		return d;
	}
	double getMaxAbsShade()
	{
		double max = 0;
		for (int i = 0; i < _width * _height; i++)
		{
			if (std::abs(shades[i]) > max)
			{
				max = std::abs(shades[i]);
			}
		}
		return max;
	}

	void normalize(){
		double max = getMaxAbsShade();
		if(max != 0){
			for (int i = 0; i < _width * _height; i++)
			{
				shades[i]/=255.0;
			}
		}
	}

	double operator[](int index) const { return shades[index]; }
	int getValue() const { return value; }
	double operator-(const digit &) const;
	friend std::istream &operator>>(std::istream &in, digit &other);

	friend std::ostream &operator<<(std::ostream &out, const digit &other);
};
class classification
{
private:
	int size;
	double *probability;
	int sampleSize;

public:
	classification(int size = 10) : size(size), sampleSize(0)
	{
		probability = new double[size];
		memset(probability, 0, size * sizeof(double));
	}
	classification(const classification &other) : size(other.size), sampleSize(other.size)
	{
		probability = new double[size];
		memcpy(probability, other.probability, sizeof(double) * other.size);
	}

	void operator=(const classification &other)
	{
		size = other.size;
		sampleSize = other.sampleSize;
		if (probability)
			delete probability;
		probability = new double[size];
		memcpy(probability, other.probability, sizeof(double) * other.size);
	}
	friend std::ostream &operator<<(std::ostream &out, const classification &other);
	~classification()
	{
		if (probability)
			delete probability;
	}

	void add(int clazz)
	{
		++sampleSize;
		probability[clazz]++;
	}

	int prediction()
	{
#ifdef safe
		normalize(); //just to be sure
#endif
		int index = 0;
		for (int i = 1; i < size; i++)
		{
			if (probability[i] > probability[index])
			{
				index = i;
			}
		}
		return index;
	}

	classification &normalize()
	{
		if (sampleSize == 0)
			return *this;
		for (int i = 0; i < size; i++)
		{
			probability[i] /= sampleSize;
		}
		return *this;
	}
};
class digitSet
{
protected:
	std::vector<digit> _digits;
	int _width;
	int _height;

public:
	digitSet(int width = 8, int height = 8) : _width(width), _height(height) {}
	void add(const digit &d)
	{
#ifdef safe
		if (d._width != _width || d._height != _height)
		{
			std << cerr << "error in digit dimension" << std::endl;
			exit(1);
		}
#endif
		_digits.push_back(d);
	}
	void normalize(){std::for_each(_digits.begin(),_digits.end(),[&](digit&digit){digit.normalize();});}
	digitSet(const digitSet &other) : _digits(other._digits), _width(other._width), _height(other._height) {}
	const digit &getDigit(int index) const { return _digits[index]; }
	const std::vector<digit> &getDigits() const { return _digits; }
	int size() const { return _digits.size(); }
	int dimension() const { return _width * _height; };
	friend std::istream &operator>>(std::istream &in, digitSet &other);
	friend std::ostream &operator<<(std::ostream &out, digitSet &other);

	friend classification kNN(const digit &check, const digitSet &training, int k);
	friend int kNNerrorFunction(const digitSet &training, const digitSet &test, int k);
	friend int minDist(const digit &d, const digitSet &set);
	friend int centroidError(const digitSet &training, const digitSet &test);
};

class centroidSet : public digitSet
{
private:
	int *occurrence_count;

public:
	centroidSet(int width = 8, int height = 8, int clazz_size = 10) : digitSet(width, height)
	{
		_digits.resize(clazz_size);
		occurrence_count = new int[clazz_size];
		memset(occurrence_count, 0, clazz_size * sizeof(int));
	}
	centroidSet(const centroidSet &other) : digitSet(other._width, other._height)
	{
		_digits = other._digits;
		occurrence_count = new int[_digits.size()];
		memcpy(occurrence_count, other.occurrence_count, _digits.size() * sizeof(int));
	}
	centroidSet &operator=(const centroidSet &other)
	{
		_digits = other._digits;
		if (occurrence_count)
			delete occurrence_count;
		memcpy(occurrence_count, other.occurrence_count, _digits.size() * sizeof(int));
		return *this;
	}
	void add(int clazz, const digit &d)
	{
#ifdef safe
		if (clazz < 0 || clazz >= _digits.size())
		{
			std::cerr << "error : wrong clazz" << clazz << std::endl;
			return -1;
		}
#endif
		occurrence_count[clazz]++;
		_digits[clazz] = _digits[clazz] + d;
	}
	void normalize()
	{
		for (int i = 0; i < _digits.size(); i++)
		{
			if (occurrence_count == 0)
				continue;
			_digits[i] = _digits[i] * (1.0 / occurrence_count[i]);
			occurrence_count[i] = 1;
		}
	}
	~centroidSet()
	{
		if (occurrence_count)
			delete occurrence_count;
	}
};

class linearRegression
{
private:
	arma::mat X;
	arma::mat Y;
	arma::mat W;
	arma::vec littlew;
	double b;
	int l;
	int signo(double number) const { return number < 0 ? -1 : 1; }

public:
	linearRegression(const digitSet &classA, const digitSet &classB) : Y(classA.size() + classB.size(), 1), l(classA.size() + classB.size())
	{
		X.set_size(l, classA.dimension() + 1);
		X.fill(1.0);
		int i = 0;
		for (auto &d : classA.getDigits())
		{

			for (int j = 0; j < classA.dimension(); j++)
			{
				X(i, j) = d[j];
			}
			Y(i, 0) = -1.0;
			++i;
		}
		for (auto &d : classB.getDigits())
		{

			for (int j = 0; j < classB.dimension(); j++)
			{
				X(i, j) = d[j];
			}
			Y(i, 0) = 1.0;
			++i;
		}
	}

	int classify(const digit &d) const
	{
		arma::vec x(d.dimension());
		for (int i = 0; i < d.dimension(); i++)
		{
			x(i) = d[i];
		}
		return signo(arma::dot(littlew, x) + b);
	}

	arma::mat getsetW(double lambda)
	{
		W = arma::inv(X.t() * X + arma::eye<arma::mat>(X.n_cols, X.n_cols) * lambda) * X.t() * Y;
		littlew = W(arma::span(0, W.n_rows - 2), arma::span(0, 0));
		b = W(W.n_rows - 1, W.n_cols - 1);
		return W;
	}
};
class crossvalidation
{
protected:
	int fold;
	std::vector<digit> _digits;

public:
	std::vector<digitSet> trainings;
	std::vector<digitSet> tests;
	crossvalidation(const digitSet &train, const digitSet &test, int fold) : fold(fold)
	{
		_digits.insert(_digits.end(), train.getDigits().begin(), train.getDigits().end());
		_digits.insert(_digits.end(), test.getDigits().begin(), test.getDigits().end());
	}
	void partition();
	virtual double meanerrorfunction(double) = 0;
};

class kNNcrossvalidation : public crossvalidation
{
public:
	kNNcrossvalidation(const digitSet &train, const digitSet &test, int fold) : crossvalidation(train, test, fold) {}
	double meanerrorfunction(double k)
	{
		double errorsum = 0;
		for (int i = 0; i < fold; i++)
		{
			errorsum += kNNerrorFunction(trainings[i], tests[i], k);
		}
		return errorsum / fold;
	}
};

digitSet filterByValue(const digitSet &, double value);
int linearRegressionErrorFunction(const linearRegression &regression, const digitSet &testSet, int classification);
class regressioncrossvalidation : public crossvalidation
{
private:
	std::vector<digitSet> testsclazzA;
	std::vector<digitSet> testsclazzB;
	std::vector<linearRegression> trainedregressions;

public:
	regressioncrossvalidation(const digitSet &train, const digitSet &test, int fold, int clazzA, int clazzB) : crossvalidation(train, test, fold)
	{
		partition();
		for (int i = 0; i < fold; i++)
		{
			testsclazzA.push_back(filterByValue(tests[i], clazzA));
			testsclazzB.push_back(filterByValue(tests[i], clazzB));
			trainedregressions.push_back(linearRegression(filterByValue(trainings[i], clazzA), filterByValue(trainings[i], clazzB)));
		}
	}
	double meanerrorfunction(double lambda)
	{
		double errorsum = 0;
		for (int i = 0; i < fold; i++)
		{
			trainedregressions[i].getsetW(lambda);
			errorsum += linearRegressionErrorFunction(trainedregressions[i], testsclazzA[i], -1);
			errorsum += linearRegressionErrorFunction(trainedregressions[i], testsclazzB[i], 1);
		}
		std::cout << "error count in regression:" << errorsum << std::endl;
		return errorsum / fold;
	}
};

__global__ void kernel(double* sample,double* map,int dim,double* distance);
class SelfOrganizingMap
{
public:
	SelfOrganizingMap(const digitSet &points, int mapWidth, int mapHeight) : _map(mapHeight, std::vector<digit>(mapWidth)), data(points), mapHeight(mapHeight), mapWidth(mapWidth)
	{
		srand(time(NULL));
		if (points.getDigits().size() < mapWidth*mapHeight )
		{
			throw "not enough data to create s.o.m!";
		}
		data.normalize();
		int digitWidth = points.getDigit(0).getWidth();
		int digitHeight = points.getDigit(0).getHeight();

		initializeSampledSOM(_map, digitWidth, digitHeight);
	}

	void initializeRandomSOM(std::vector<std::vector<digit>> &som, int dimx, int dimy)
	{
		std::for_each(som.begin(), som.end(), [dimx, dimy](std::vector<digit> &v) {
			std::fill(v.begin(), v.end(), digit(dimx, dimy));

			std::for_each(v.begin(), v.end(), [](digit &d) {
				d.initrandom();
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

	void train(int maxT,std::function<void(int,int,SelfOrganizingMap*)> &&f=[](int,int,SelfOrganizingMap*){})
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
			f(T,maxT,this);
			randomSampleIndex = rand() % data.getDigits().size();
			closestPrototype = getClosestPrototypeIndices(data.getDigits()[randomSampleIndex]);

			maxAdjusted = updateNeighbours(closestPrototype, learningRate, neighbourRadius, data.getDigits()[randomSampleIndex]);
			learningRate = initiallearningrate * normal_pdf(T, 0, maxT / 3);
			neighbourRadius = std::max(0.05, (std::max(mapWidth, mapHeight) / windowSmallness) * (maxT - T) / maxT /*normal_pdf( T, 0, maxT)*/);

			T++;
		}

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
		out << mapHeight << " " << mapWidth << std::endl;
		for (int i = 0; i < mapHeight; i++)
		{
			for (int j = 0; j < mapWidth; j++)
			{
				_map[i][j].appendToFile(out);
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

	
	
	std::pair<int, int> getClosestPrototypeIndices(const digit &sample)
	{
		//  if(mapWidth < 65535 && mapHeight < 65535){
			int dim = sample.dimension();
			double* dev_distance; 
			cudaMalloc((void**)&dev_distance,sizeof(double)*mapWidth*mapHeight);
			// mapwidth * mapheight
			double* dev_sample;
			cudaMalloc((void**)&dev_sample,sizeof(double)*dim);
			cudaMemcpy((double*)dev_sample,sample.getShades(),sizeof(double)*dim,cudaMemcpyHostToDevice);
			// dim
			double* dev_map;
			cudaMalloc((void**)&dev_map,sizeof(double)*dim* mapWidth * mapHeight);
			// dim * mapwidth * mapheight
			for(int i=0;i<mapHeight;i++){
				for(int j=0;j<mapWidth;j++){
					cudaMemcpy((double*)(dev_map + i*mapWidth*dim + j*dim),_map[i][j].getShades(),sizeof(double)*dim,cudaMemcpyHostToDevice);
				}
			}
			
			kernel<<<mapWidth*mapHeight,1>>>(dev_sample,dev_map,dim,dev_distance);

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
					std::cout<<" current distance"<<*(distances + i*mapWidth +j)<<" minDistance "<<cuminDist<<std::endl;
				}
			}
			std::cout<<" ----- "<<std::endl;
			cudaFree(dev_distance);
			cudaFree(dev_sample);
			cudaFree(dev_map);	
			if(maxcuda_i >= mapHeight || maxcuda_j >= mapWidth){
				std::cerr<<"will segfault"<<std::endl;
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
	int mapHeight;
	int mapWidth;
};
#endif
