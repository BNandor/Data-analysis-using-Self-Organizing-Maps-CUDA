#ifndef OCRDIGITSET_H
#define OCRDIGITSET_H

#include <algorithm>
#include <fstream>
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include "digit.cu"

class digitSet {
protected:
    std::vector<digit> _digits;
    int _width;
    int _height;

public:
    digitSet(int width = 8, int height = 8)
        : _width(width)
        , _height(height)
    {
    }

    void add(const digit& d)
    {
#ifdef safe
        if (d._width != _width || d._height != _height) {
            std << cerr << "error in digit dimension" << std::endl;
            exit(1);
        }
#endif
        _digits.push_back(d);
    }

    std::pair<double, double> minMaxFeatureScale()
    {
        double minimum = std::numeric_limits<double>::max();
        std::for_each(_digits.begin(), _digits.end(), [&](digit& digit) {
            double digitMinshade = digit.getMinShade();
            if (minimum > digitMinshade) {
                minimum = digitMinshade;
            }
        });
        double maximum = std::numeric_limits<double>::min();
        std::for_each(_digits.begin(), _digits.end(), [&](digit& digit) {
            double digitMaxshade = digit.getMaxShade();
            if (maximum < digitMaxshade) {
                maximum = digitMaxshade;
            }
        });
        std::for_each(_digits.begin(), _digits.end(), [&](digit& digit) {
            digit.minMaxNormalize(minimum, maximum);
        });
        return std::make_pair(minimum, maximum);
    }

    digitSet(const digitSet& other)
        : _digits(other._digits)
        , _width(other._width)
        , _height(other._height)
    {
    }
    const digit& getDigit(int index) const { return _digits[index]; }

    const std::vector<digit>& getDigits() const { return _digits; }

    int size() const { return _digits.size(); }

    int dimension() const { return _width * _height; };

    friend std::istream& operator>>(std::istream& in, digitSet& other)
    {
        std::string line;
        while (getline(in, line)) {
            std::stringstream ss(line);
            digit d(other._width, other._height);
            ss >> d;
            other._digits.push_back(d);
        }
        return in;
    }

    friend std::ostream& operator<<(std::ostream& out, digitSet& other)
    {
        for (int i = 0; i < other._digits.size(); i++) {
            out << other._digits[i] << std::endl;
        }
        return out;
    }
};

digitSet filterByValue(const digitSet& tofilter, double value)
{
    digitSet filtered;
    for (int i = 0; i < tofilter.size(); i++) {
        if (tofilter.getDigit(i).getValue() == value) {
            filtered.add(tofilter.getDigit(i));
        }
    }
    return filtered;
}

#endif