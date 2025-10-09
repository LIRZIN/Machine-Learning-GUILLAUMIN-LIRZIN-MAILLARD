#include "exemple_Rosenblatt.hpp"

void ExempleRosenblatt::populate_points_random( float begin_range, float end_range, bool is_whole )
{
    for( int i = 0; i < nb_points; i++ )
    {
        Point point;
        point.generate_random( begin_range, end_range, is_whole );
        points.push_back( point );
    }
}

void ExempleRosenblatt::generate_weights()
{
    for( int i = 0; i < nb_weights; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights.push_back( weight * 2.0 - 1.0 );
    }
}

void ExempleRosenblatt::init_classify( std::string class1Color, std::string class2Color )
{
    for( int i = 0; i < nb_points; i++ )
    {
        Point& point = points[i];
        Color color;
        float class_value;

        if( point.getY() >= a*point.getX() + b )
        {
            color.set(class1Color);
            class_value = 1.0;
        }
        else 
        {
            color.set(class2Color);
            class_value = -1.0;
        }

        colors.push_back( color );
        class_values.push_back( class_value );
    }
}