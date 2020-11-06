#include "ocr.cu"
#include <math.h>
#include <iostream>
#include <sstream>
#include <queue>
#include <algorithm>
#include "parallel.h"
#include <limits.h>
#include <armadillo>
#define thread_num 10

digit::digit(int width,int height):_width(width),_height(height){
	shades = new double[width*height];
	memset(shades,0,(width*height)*sizeof(int));
}

digit::~digit(){
	if(shades){
		delete shades;	}
}

double digit::operator -(const digit & other)const{
#ifdef safe
	if(_width != other._width || _height != other.height){
		cout<<"error, different dimensions"<<endl;
		exit(0);
	}
#endif	

	double sum=0;
	for(int i=0;i<_width*_height;i++){
		sum+=pow(shades[i]-other.shades[i],2);
	}
#ifdef squareDistance
	return sum;
#else
	return sqrt(sum);
#endif	
}
std::ostream& operator<<(std::ostream& out,const digit& other){
	for (int i = 0; i < other._width* other._height; i++) {
		out<<other.shades[i]<<" ";
	}
	out<<"->"<<other.value;
	return out;
}

std::istream& operator >>(std::istream & in,digit& other){
	for (int i = 0; i <other._width*other._height; i++) {
		in>>other.shades[i];
	}
	in>>other.value;
	return in;
}

std::istream& operator >>(std::istream & in,digitSet& other){
	std::string line;
	while(getline(in,line)){
		std::stringstream ss(line);		
		digit d(other._width,other._height);
		ss>>d;
		other._digits.push_back(d);
	}
	return in;
}
std::ostream& operator <<(std::ostream & out,digitSet& other){
	for (int i = 0; i < other._digits.size(); i++) {
		out<<other._digits[i]<<std::endl;	
	}
	return out;
}
std::ostream& operator<<(std::ostream& out,const classification & other){
	for (int i = 0; i < other.size; i++) {
		out<<other.probability[i]<<" ";
	}
	return out;
}
bool bigger(std::pair<int, double> a, std::pair<int, double>b ){
	return a.second < b.second;
}

classification kNN(const digit& check,const digitSet& training,int k){
	std::vector<std::pair<int ,double > > distances;
	int i=0;
	for (auto & d:training._digits) {
		distances.push_back(std::make_pair(training._digits[i].getValue(),check-d));
		++i;
	}//sort in increasing order according to distance
	sort(distances.begin(),distances.end(),bigger);
	classification c;
	

	for (int i = 0; i < k; i++) {
		c.add(distances[i].first);
	}	
	return c.normalize();
}

struct job{
	const digit * test;
	const digitSet * training;
	int k;	
	concurrent::Output<int> *output;
};

struct kNNFunctor {
	void operator()(job & param)const{
			
		if(kNN( *param.test, *param.training, param.k).prediction()!=param.test->getValue()){
			param.output->push(1);
		}else{
			param.output->push(0);
		}
	}
	
};

int kNNerrorFunction(const digitSet& training, const digitSet& test, int k){
	int errors=0;
	for (auto & d:test._digits) {
	}

	std::queue<job> jobs;
	struct job jobdata;
	concurrent::Output<int> output;
	for (int i = 0; i < test._digits.size(); ++i)
	{
		jobdata.test = & (test._digits[i]);				
		jobdata.training = & training;
		jobdata.k = k;
		jobdata.output=&output;	
		jobs.push(jobdata);
	}
	

	kNNFunctor summer;
	concurrent::PARALLEL<kNNFunctor,job> magic(summer,jobs,thread_num);
	magic.pstart();
	
	while(output.size() > 0){
		errors+=output.pop();
	}
	return errors;
}

int minDist(const digit & d, const digitSet& set){
	double mindist = INT_MAX;
	int index;
	for (int i = 0; i < set.size(); i++) {
		if( abs(d - set.getDigit(i))<mindist)	{
			mindist=abs(d - set.getDigit(i));
			index=i;
		}
	}
	return index;
}


int centroidError( const digitSet& training , const digitSet& test){
	int error=0;
	centroidSet cset;
	for (auto & d: training.getDigits()) {
		cset.add(d.getValue(),d);
	}
	
	cset.normalize();
	for (int i = 0; i < test.size(); i++) {
		if( minDist(test.getDigit(i),cset)!=test.getDigit(i).getValue())
			error++;
	}
	return error;	
}

digitSet filterByValue(const digitSet& tofilter,double value){
	digitSet filtered;
	for (int i = 0; i < tofilter.size(); i++) {
		if(tofilter.getDigit(i).getValue() == value){
			filtered.add(tofilter.getDigit(i));
		}
	}
	return filtered;	
}
int linearRegressionErrorFunction(const linearRegression& regression,const digitSet& testSet, int classification){
		int error=0;
		for (int i = 0; i < testSet.size(); i++) {
			if(regression.classify(testSet.getDigit(i))!=classification)
				error++;
		}
return error;		
}

void crossvalidation::partition(){
		for (int i = 0; i <fold; i++) {
			digitSet dtr;
			trainings.push_back(dtr);
			digitSet dte;
			tests.push_back(dte);
		}
		int window = _digits.size()/fold;
		for (int i = 1; i <= fold; i++) {
			for (int j = (i-1)*window; j <i*window ; j++) {
				tests[i-1].add(_digits[j]);
				for (int l = 0; l < fold; l++) {
					if(l!=i-1){
						trainings[l].add(_digits[j]);
					}
				}
			}
		}
}
//double* sample,double* map,int dim,double* distance
__global__ void kernel(double* sample,double* map,int dim, double* distance){
	int offset = blockIdx.x;
	double* protoype = map + offset*dim;
	double sum = 0;
	for(int c=0;c<dim;c++){
		sum+=(sample[c]-protoype[c])* (sample[c]-protoype[c]);
	}
	distance[offset] = sqrt(sum);
}