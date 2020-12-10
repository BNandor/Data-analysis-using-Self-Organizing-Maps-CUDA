#ifndef OCRDIGIT_H
#define OCRDIGIT_H
#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
/** \brief A feature vector container of continuous features. 
 * */
/** 
 * This class is a container of continuous features. 
 * It supports +,- vector operations, random initialization within a range, and min/max normalization.
 * It also contains a classification label.
 * */
class digit {
private:
    int value;
    double* shades;
    int _width;
    int _height;

public:
    /** \brief Return the number of features in the vector.*/
    int dimension() const { return _width * _height; }
    /** \brief The constructor takes as argument a width and height of the vector. */
    digit(int width = 8 /**< The width of the image.*/, int height = 8 /**< The height of the image.*/)
        : _width(width)
        , _height(height)
    {
        shades = new double[width * height];
        memset(shades, 0, (width * height) * sizeof(int));
    }

    /** \brief Deallocates the feature vector. */
    ~digit()
    {
        if (shades) {
            delete shades;
        }
    }
    /** \brief Copies the feature values from the other container into a newly allocated memory region.*/
    digit(const digit& other)
    {
        shades = new double[other._width * other._height];
        memcpy(shades, other.shades,
            sizeof(double) * (other._width * other._height));
        _width = other._width;
        _height = other._height;
        value = other.value;
    }

    /** Copies the feature values from the other container into a newly allocated memory region. 
     * Deallocates the old memory to enable resizing.
     * */
    void operator=(const digit& other)
    {
        if (shades)
            delete shades;
        shades = new double[other._width * other._height];
        memcpy(shades, other.shades,
            sizeof(double) * (other._width * other._height));
        _width = other._width;
        _height = other._height;
        value = other.value;
    }

    /** \brief Returns the width of the container.*/
    int getWidth() const { return _width; }

    /** \brief Returns the height of the container.*/
    int getHeight() const { return _height; }

    /** \brief Returns the pointer to the allocated memory region.*/
    double* getShades() const { return shades; }

    /** \brief Randomly initializes the values in the container in a certain range.*/
    void initrandom(double min /**<The lowest possible random value.*/,
        double max /**<The highest possible random value.*/)
    {
        for (int i = 0; i < dimension(); i++) {
            shades[i] = min + ((double)rand() / (RAND_MAX)) * (max - min);
        }
    }
    /** Prints the features to an output stream in lexicographical order, followed by an endline.
     * It supports a mapping functionality as well.
    */
    std::ostream& appendToFile(
        std::ostream& out /**<The output stream to print the features to.*/,
        std::function<double(double)>&& mappedShade = [](double shade) { return shade; } /**< The mapping function of the features. 
        It defaults to the identity function. */
    )
    {
        for (int i = 0; i < _width * _height; i++) {
            out << mappedShade(shades[i]) << " ";
        }
        out << std::endl;
        return out;
    }

    /** \brief Return a new digit that is the vectorial sum of the instance and the other vector.*/
    digit operator+(const digit& other) const
    {
#ifdef safe
        if (_width != other._width || _height != other._height) {
            std::cerr << "dimension discrepancy" << std::endl;
            return -1;
        }
#endif
        digit sum(*this);
        for (int i = 0; i < _width * _height; i++) {
            sum.shades[i] += other.shades[i];
        }
        return sum;
    }

    /** \brief Return a new digit that is the vectorial subtraction of the instance and the other vector.*/
    digit minus(const digit& other) const
    {
#ifdef safe
        if (_width != other._width || _height != other._height) {
            std::cerr << "dimension discrepancy" << std::endl;
            return -1;
        }
#endif
        digit difference(*this);
        for (int i = 0; i < _width * _height; i++) {
            difference.shades[i] -= other.shades[i];
        }
        return difference;
    }

    /** \brief Return a new digit that is the  scalar multiplication of the instance with the provided scalar.*/
    digit operator*(double scalar /**<The scalar with which the instance is multiplied.*/) const
    {
        digit d(*this);
        for (int i = 0; i < _width * _height; i++) {
            d.shades[i] *= scalar;
        }
        return d;
    }

    /** \brief Returns the maximum absolut value of the features of the container.*/
    double getMaxAbsShade()
    {
        double max = 0;
        for (int i = 0; i < _width * _height; i++) {
            if (std::abs(shades[i]) > max) {
                max = std::abs(shades[i]);
            }
        }
        return max;
    }

    /**  \brief Returns the maximum  value of the features of the container.*/
    double getMaxShade()
    {
        double max = std::numeric_limits<double>::min();
        for (int i = 0; i < _width * _height; i++) {
            if (shades[i] > max) {
                max = shades[i];
            }
        }
        return max;
    }

    /** \brief Returns the minimum absolut value between the features of the container.*/
    double getMinShade()
    {
        double min = _width != 0 && _height != 0 ? shades[0] : 0;
        for (int i = 1; i < _width * _height; i++) {
            if (shades[i] < min) {
                min = shades[i];
            }
        }
        return min;
    }

    /** \brief Normalizes the features using min,max normalization.*/
    void minMaxNormalize(double min /**< The bottom of the normalization range.*/,
        double max /**< The top of the normalization range.*/)
    {
        if (max <= min) {
            throw "[DIGIT:minMaxNormalize] Error, max <= min!";
        }

        double width = (max - min);
        for (int i = 0; i < _width * _height; i++) {
            shades[i] = (shades[i] - min) / width;
        }
    }

    /** \brief Returns the feature in the container at a particular index.*/
    double operator[](int index /**< The index of the feature to be returned.*/) const { return shades[index]; }

    /** \brief Returns the index of the label of the current vector. */
    int getValue() const { return value; }

    /** \brief Return the distance between two vectors in euclidean distance metric.*/
    double operator-(const digit& other) const
    {
#ifdef safe
        if (_width != other._width || _height != other.height) {
            cout << "error, different dimensions" << endl;
            exit(0);
        }
#endif

        double sum = 0;
        for (int i = 0; i < _width * _height; i++) {
            sum += pow(shades[i] - other.shades[i], 2);
        }
#ifdef squareDistance
        return sum;
#else
        return sqrt(sum);
#endif
    }

    /** \brief Friend function, reads a new feature vector from an input stream.*/
    friend std::istream& operator>>(std::istream& in /**< The input stream from which the features will be read. 
    After the features the associated label is read.*/
        ,
        digit& other /**< The container reference that will contain the read features.*/)
    {
        for (int i = 0; i < other._width * other._height; i++) {
            in >> other.shades[i];
        }
        in >> other.value;
        return in;
    }

    /** \brief Prints the features to an output stream and the associated label index.*/
    friend std::ostream& operator<<(std::ostream& out /**< The output stream to which the features will be written. 
    After the features the associated label is also printed.*/
        ,
        const digit& other)
    {
        for (int i = 0; i < other._width * other._height; i++) {
            out << other.shades[i] << " ";
        }
        out << "->" << other.value;
        return out;
    }
};
#endif