cmake_minimum_required(VERSION 3.22)

# Nom de la librairie
project(ML_Lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclure Eigen (header-only)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-5.0.0
)

# Tous les fichiers sources de la librairie
set(SOURCES
    LinearModel.cpp
    LinearModel.hpp
    MLP.cpp
    MLP.hpp
    RBF.cpp
    RBF.hpp
    ML_c_wrapper.cpp
)

# Créer la DLL
add_library(ML_Lib SHARED ${SOURCES})
set_target_properties(ML_Lib PROPERTIES PREFIX "")

# Créer un exécutable pour tester localement
add_executable(ML_Test main.cpp LinearModel.cpp MLP.cpp RBF.cpp)

