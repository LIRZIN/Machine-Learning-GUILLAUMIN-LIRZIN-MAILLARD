#ifndef EXEMPLE_ROSENBLATT_HPP
#define EXEMPLE_ROSENBLATT_HPP

#include "Point.hpp"
#include "Color.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class ExempleRosenblatt 
{
    int nb_points;
    std::vector<Point> points;
    std::vector<Color> colors;
    std::vector<float> class_values;

    int nb_weights;
    Eigen::VectorXd weights;

    int MSE_interval;
    std::vector<float> MSEs;
    
    float a, b, alpha; // y = a * x + b

    public : 
        ExempleRosenblatt( int nb_points, int nb_weights, float a, float b, float  alpha ) : nb_points(nb_points), nb_weights(nb_weights), a(a), b(b), alpha(alpha) {}

        // INITIALIZATION
        void populate_points_random( float begin_range = 0.0, float end_range = 1.0, bool is_whole = false );
        void init_classify( std::string class1Color = "blue", std::string class2Color = "red" );

        // MODEL CREATION
        void generate_weights();
        float predict( Eigen::VectorXd& X_k_with_one );

        // MODEL TRAINING USING ROSENBLATT'S RULE
        void train( int nb_iterations, int interval = 0 );

        // PRINT FUNCTIONS
        void print_points( std::string pathToFile );
        void print_background( std::string pathToFile, std::string class1BG = "lightblue", std::string class2BG = "pink" );
        void print_MSE( std::string pathToFile, bool print_index = true );
};

#endif