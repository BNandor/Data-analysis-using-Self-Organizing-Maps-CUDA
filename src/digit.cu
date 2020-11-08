#ifndef OCRDIGIT_H
#define OCRDIGIT_H
#include <fstream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits>
#include <functional>

class digit
{
private:
	int value;
	double *shades;
	int _width;
	int _height;

public:
	int dimension() const { return _width * _height; }
	digit(int width = 8, int height = 8):_width(width),_height(height){
		shades = new double[width*height];
		memset(shades,0,(width*height)*sizeof(int));
	}

	~digit(){
		if(shades){
		delete shades;	}
	
	}
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
    
	void initrandom(double min, double max)
	{
		for (int i = 0; i < dimension(); i++)
		{
			shades[i] = min + ((double) rand() / (RAND_MAX)) * (max - min);
		}
	}

	std::ostream &appendToFile(std::ostream &out,std::function<double(double)> &&mappedShade=[](double shade){return shade;})
	{
		for (int i = 0; i < _width * _height; i++)
		{
			out << mappedShade(shades[i]) << " ";
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
    
	double getMaxShade()
	{
		double max=std::numeric_limits<double>::min();
		for (int i = 0; i < _width * _height; i++)
		{
			if (shades[i] > max)
			{
				max = shades[i];
			}
		}
		return max;
    }
    

    double getMinShade()
	{
		double min = _width !=0 && _height != 0 ? shades[0]:0;
		for (int i = 1; i < _width * _height; i++)
		{
			if (shades[i] < min)
			{
				min = shades[i];
			}
		}
		return min;
    }

    void minMaxNormalize(double min, double max){

        if(max <= min){
            throw "[DIGIT:minMaxNormalize] Error, max <= min!";
        }

        double width = (max - min);
        for (int i = 0; i < _width * _height; i++)
		{
            shades[i]= (shades[i]-min) / width;
        }
    }

	double operator[](int index) const { return shades[index]; }
	int getValue() const { return value; }
	double operator-(const digit & other) const{
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

	friend std::istream &operator>>(std::istream &in, digit &other){
		for (int i = 0; i <other._width*other._height; i++) {
			in>>other.shades[i];
		}
		in>>other.value;
		return in;
	}
	

	friend std::ostream &operator<<(std::ostream &out, const digit &other){
		for (int i = 0; i < other._width* other._height; i++) {
			out<<other.shades[i]<<" ";
		}
		out<<"->"<<other.value;
		return out;
	}
	
};
#endif