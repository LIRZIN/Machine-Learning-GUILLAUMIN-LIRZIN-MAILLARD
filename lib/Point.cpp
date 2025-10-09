#include "Point.hpp"

void Point::generate_random( float begin_range, float end_range, bool is_whole = false )
{
    float r_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float r_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    x = begin_range + r_x * ( end_range - begin_range );
    y = begin_range + r_y * ( end_range - begin_range );

    if( is_whole )
    {
        x = std::round(x);
        y = std::round(y);
    }
}