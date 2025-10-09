#include "Color.hpp"

void Color::set( float new_R, float new_G, float new_B )
{
    setR(new_R);
    setG(new_G);
    setB(new_B);
}

void Color::set( std::string color )
{
    switch( color )
    {
        case "red" : set(255, 0, 0); break;
        case "green" : set(0, 255, 0); break;
        case "blue" : set(0, 0, 255); break;
        default : set(0, 0, 0);
    }
}