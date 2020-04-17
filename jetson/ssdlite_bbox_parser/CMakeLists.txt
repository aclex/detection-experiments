cmake_minimum_required(VERSION 3.13)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc" CACHE STRING "CUDA compiler")

project(ssdlite_bbox_parser LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

set(DEEPSTREAM_SDK_PATH "/opt/nvidia/deepstream/deepstream-4.0" CACHE STRING "DeepStream SDK path")

find_package(TensorRT REQUIRED)
find_package(DeepStream REQUIRED)
