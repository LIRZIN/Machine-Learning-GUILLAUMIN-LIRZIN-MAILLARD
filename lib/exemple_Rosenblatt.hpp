#ifndef EXEMPLE_ROSENBLATT_HPP
#define EXEMPLE_ROSENBLATT_HPP

#include "Point.hpp"
#include "Color.hpp"
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

class ExempleRosenblatt 
{
    int nb_points;
    std::vector<Point> points;
    std::vector<Color> colors;
    std::vector<float> class_values;

    int nb_weights;
    std::vector<float> weights;
    
    float a, b; // y = a * x + b

    public : 
        ExempleRosenblatt( int nb_points, int nb_weights, float a_linear_equ, float b_linear_equ )
        {
            nb_points = nb_points;

            nb_weights = nb_weights;

            a = a_linear_equ;
            b = b_linear_equ;
        }

        // INITIALIZATION
        void populate_points_random( float begin_range = 0.0, float end_range = 1.0, bool is_whole = false );
        void generate_weights();
        void init_classify( std::string class1Color = "blue", std::string class2Color = "red" );

        // MODEL CREATION


        // MODEL TRAINING USING ROSENBLATT'S RULE
        
};

#endif