#ifndef COLOR_HPP
#define COLOR_HPP

#include <string>

class Color
{
    unsigned char R;
    unsigned char G; 
    unsigned char B;

    public : 
        Color( unsigned char init_R = 0, unsigned char init_G = 0, unsigned char init_B = 0 ) : R(init_R), G(init_G), B(init_B) {}
        Color( std::string color ) { set( color ); }

        void setR( float val ) { R = val; }
        void setG( float val ) { G = val; }
        void setB( float val ) { B = val; }
        void set( float new_R, float new_G, float new_B );
        void set( std::string color );

        float getR() { return R; }
        float getG() { return G; }
        float getB() { return B; }
};

#endif