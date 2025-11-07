#include "MLP.hpp"

int main()
{
    /*
    int layers[2] = { 2, 1 };
    MLP mlp(2, layers);

    mlp.initElements( 3 );
    mlp.addElement( 3, 1.0, 1.0, 1.0 );
    mlp.addElement( 3, 2.0, 3.0, -1.0 );
    mlp.addElement( 3, 3.0, 3.0, -1.0 );
    
    mlp.train( 100000, 0.01, true );

    std::cout << "success rate : " << mlp.test( true ) << "%\n";
    */
    /*
    int layers[3] = { 2, 3, 1 };
    MLP mlp(3, layers);

    mlp.initElements( 4 );
    mlp.addElement( 3, 0, 0, -1 );
    mlp.addElement( 3, 1, 0, 1 );
    mlp.addElement( 3, 0, 1, 1 );
    mlp.addElement( 3, 1, 1, -1 );
    
    mlp.train( 1000000, 0.01, true );

    std::cout << "success rate : " << mlp.test( true ) << "%\n";
    */
    int layers[3] = { 2, 1 };
    MLP mlp(2, layers);
    int nb_elements = 200;
    mlp.initElements( nb_elements );

    for( int i = 0; i < nb_elements; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y1 = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;
        //double y2 = ( x2 >= -1.0*x1 + 0.7 ) ? -1.0 : 1.0;
        mlp.addElement( 3, x1, x2, y1 );
    }
    
    mlp.train( 100000, 0.01, true );
    float array[2] = { 0.5, 0.5 };
    mlp.generatePredictionArray( true, 2, array );

    std::cout << "success rate : " << mlp.test( true ) << "%\n";
}