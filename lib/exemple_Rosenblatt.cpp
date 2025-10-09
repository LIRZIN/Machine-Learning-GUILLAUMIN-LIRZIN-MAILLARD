#include "exemple_Rosenblatt.hpp"

void ExempleRosenblatt::populate_points_random( float begin_range = 0.0, float end_range = 1.0, bool is_whole = false )
{
    for( Point& point : points )
    {
        point.generate_random( begin_range, end_range, is_whole );
    }
}

void ExempleRosenblatt::generate_weights()
{
    for( float& weight : weights )
    {
        weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weight = weight * 2.0 - 1.0;
    }
}

void ExempleRosenblatt::init_classify( std::string class1Color = "blue", std::string class2Color = "red" )
{
    for( int i = 0; i < nb_points; i++ )
    {
        Point& point = points[i];
        Color& color = colors[i];
        float& class_value = class_values[i];

        if( point.getY() >= a*point.getX() + b )
        {
            color = class1Color;
            class_value = 1.0;
        }
        else 
        {
            color = class2Color;
            class_value = -1.0;
        }
    }
}