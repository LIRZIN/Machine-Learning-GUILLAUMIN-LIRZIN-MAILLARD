#ifndef POINT_HPP
#define POINT_HPP

#include <cstdlib>
#include <cmath>

class Point
{
    float x;
    float y;

    public : 
        Point( float init_x = 0, float init_y = 0 ) : x(init_x), y(init_y) {}

        void setX( float val ) { x = val; }
        void setY( float val ) { y = val; }

        float getX() { return x; }
        float getY() { return y; }

        void generate_random( float begin_range, float end_range, bool is_whole );
};

#endif