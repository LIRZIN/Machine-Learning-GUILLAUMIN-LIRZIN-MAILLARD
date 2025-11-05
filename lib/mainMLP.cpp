#include "MLP.hpp"

int main()
{
    int layers[2] = { 2, 1 };
    MLP mlp(2, layers);
    int nb_elements = 100;
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
    mlp.printElements();

    double success = 0.0;

    for( int i = 0; i < 100; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y1 = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;
        //double y2 = ( x2 >= -1.0*x1 + 0.7 ) ? -1.0 : 1.0;

        mlp.generatePrediction( true, 2, x1, x2 );

        double res0 = mlp.getPrediction(0);
        //double res1 = mlp.getPrediction(1);
        //std::cout << res0 << ", " << res1 << "||" << y1 << ", " << y2 << std::endl;

        std::cout << res0 << "||" << y1 << std::endl;

        if( ( y1 < 0 && res0 < 0 ) || ( y1 > 0 && res0 > 0 ) )
        {
            success += 1.0;
        }
        /*
        if( ( y1 < y2 && res0 < res1 ) || ( y1 > y2 && res0 > res1 ) )
        {
            success += 1.0;
        }
        */
    }

    std::cout << "success rate : " << success << "%\n";
}