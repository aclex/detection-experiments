cmake_minimum_required(VERSION 3.10)

project(bbox_parser LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

set(DEEPSTREAM_SDK_PATH "/opt/nvidia/deepstream/deepstream-4.0" CACHE STRING "DeepStream SDK path")

find_package(DeepStream REQUIRED)

add_subdirectory(src)
