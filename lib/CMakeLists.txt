cmake_minimum_required(VERSION 3.22)
project(Multi_Level_Perceptron_Lib)

set(CMAKE_CXX_STANDARD 20)

add_library(Multi_Level_Perceptron_Lib SHARED MultiLayerPerceptron.cpp MultiLayerPerceptron.hpp )
target_include_directories(Multi_Level_Perceptron_Lib PRIVATE ${CMAKE_SOURCE_DIR}/eigen-5.0.0 )