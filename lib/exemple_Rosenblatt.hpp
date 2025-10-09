#ifndef EXEMPLE_ROSENBLATT_HPP
#define EXEMPLE_ROSENBLATT_HPP

#include "Point.hpp"
#include "Color.hpp"
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
            self->nb_points = nb_points;
            points.resize( nb_points );
            colors.resize( nb_points );
            class_values.resize( nb_points );

            self->nb_weights = nb_weights;
            weights.resize( nb_weights );

            a = a_linear_equ;
            b = b_linear_equ;
        }

        void populate_points_random( float begin_range = 0.0, float end_range = 1.0, bool is_whole = false );
        void generate_weights();
        void init_classify( std::string class1Color = "blue", std::string class2Color = "red" );
        
};

#endif