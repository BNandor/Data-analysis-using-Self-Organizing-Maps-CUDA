cmake_minimum_required(VERSION 3.7)
project(OCR)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/confusion_matrix.cu
        src/crossvalidation.cu
        src/cuda.cu
        src/digit.cu
        src/digits.cu
        src/digitSet.cu
        src/io.cu
        src/Makefile
        src/SOM.cu)

include_directories(src)

add_executable(OCR ${SOURCE_FILES})