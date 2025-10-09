#include "Color.hpp"

void Color::set( float new_R, float new_G, float new_B )
{
    setR(new_R);
    setG(new_G);
    setB(new_B);
}

void Color::set( std::string color )
{
    if( color == "red" )              { set(255, 0, 0); }
    else if( color == "green" )       { set(0, 255, 0); }
    else if( color == "blue" )        { set(0, 0, 255); }
    else                              { set(0, 0, 255); }
}