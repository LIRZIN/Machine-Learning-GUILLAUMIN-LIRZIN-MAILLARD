cmake_minimum_required(VERSION 3.22)
project(_2025_3A_IABD1_DemoInteropCpp)

set(CMAKE_CXX_STANDARD 20)

add_library(_2025_3A_IABD1_DemoInteropCpp SHARED Color.cpp Color.hpp exemple_Rosenblatt.hpp exemple_Rosenblatt.cpp Point.hpp Point.cpp )
target_include_directories(_2025_3A_IABD1_DemoInteropCpp PRIVATE ${CMAKE_SOURCE_DIR}/eigen-5.0.0 )