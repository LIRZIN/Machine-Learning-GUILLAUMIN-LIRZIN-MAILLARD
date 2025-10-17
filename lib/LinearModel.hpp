#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

class LinearModel
{
    // n-th column : n-th element
    // n-th line : n-th component of an element
    // Eigen::MatrixXd<type, nb_rows, nb_columns>
    // typedef Matrix<float, Dynamic, 1> VectorXf;
    int nb_components_of_element;
    int nb_elements = 0;

    Eigen::MatrixXd input;
    Eigen::VectorXd output;
    Eigen::VectorXd weights;

    /*
    int MSE_interval;
    std::vector<float> MSEs;
    */

    public : 
        LinearModel( int init_nb_components_of_element = 2 );

        void addElement( int count, ... );
        void printElements();

        double predict( Eigen::VectorXd& X_k_with_one );
        void train( int nb_iterations, double alpha );
};

#endif