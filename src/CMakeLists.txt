cmake_minimum_required(VERSION 3.17)
project(src CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(src main.cu)

set_target_properties(
        src
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)