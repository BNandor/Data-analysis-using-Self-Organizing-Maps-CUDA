#ifndef OCRDIGITSET_H
#define OCRDIGITSET_H

#include "digit.cu"
#include <algorithm>
#include <fstream>
#include <limits>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

/** This class is a container for a set of vector samples.
 *  It supports the addition of elements, sample normalization and readind writing to streams.
*/
class digitSet {
protected:
    std::vector<digit> _digits;
    int _width;
    int _height;

public:
    /** \brief The constructor takes as argument the dimensionality of the vectors. */
    digitSet(int width, int height = 1)
        : _width(width)
        , _height(height)
    {
    }
    /** \brief Adds a new sample to the set of samples.*/
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

    /**  \brief This method normalizes the samples over the maximum and minimum value of the overall features.*/
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

    /** \brief Returns the width of the feature vectors.*/
    int getWidth() const { return _width; }

    /** \brief Returns the height of the feature vectors.*/
    int getHeight() const { return _height; }

    /** \brief Returns the digit at a given index.*/
    const digit& getDigit(int index) const { return _digits[index]; }

    /** \brief Returns container that holds the samples.*/ 
    const std::vector<digit>& getDigits() const { return _digits; }

    /** \brief Returns the number of samples held in the container.*/
    int size() const { return _digits.size(); }

    /** \brief Returns the overall dimensionality of every sample. Every sample is of the same dimensionality.*/
    int dimension() const { return _width * _height; };

    /** Reads a sample set from an input stream. It parses the vectors line by line until end of stream.*/
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

    /** Prints a sample set to an output stream line by line.*/
    friend std::ostream& operator<<(std::ostream& out, digitSet& other)
    {
        for (int i = 0; i < other._digits.size(); i++) {
            out << other._digits[i] << std::endl;
        }
        return out;
    }

    /** \brief Filters all the samples of a particular class from a sample set into a new set.*/
    static digitSet filterByValue(const digitSet& tofilter, double value)
    {
        digitSet filtered(tofilter.getWidth(), tofilter.getHeight());
        for (int i = 0; i < tofilter.size(); i++) {
            if (tofilter.getDigit(i).getValue() == value) {
                filtered.add(tofilter.getDigit(i));
            }
        }
        return filtered;
    }
};


#endif