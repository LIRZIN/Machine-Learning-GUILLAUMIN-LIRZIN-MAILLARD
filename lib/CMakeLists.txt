cmake_minimum_required(VERSION 3.22)
project(ML_Lib)

set(CMAKE_CXX_STANDARD 17)

add_library(ML_Lib SHARED LinearModel.cpp LinearModel.hpp MLP.cpp MLP.hpp RBF.cpp RBF.hpp )
target_include_directories(ML_Lib PRIVATE ${CMAKE_SOURCE_DIR}/eigen-5.0.0 )