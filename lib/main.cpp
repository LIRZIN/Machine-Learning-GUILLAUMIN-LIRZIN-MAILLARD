//#include "exemple_Rosenblatt.hpp"
//#include "LinearModel.hpp"
#include "MultiLevelPerceptron.hpp"

int main()
{
    srand( time(0) );

    MultiLevelPerceptron mlp(2, 1);

    for( int i = 0; i < 100; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;
        mlp.addElement( 3, x1, x2, y );
    }
    
    mlp.train( 10000, 0.001 );

    double success = 0.0;

    for( int i = 0; i < 100; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;

        double res = mlp.predict( x1, x2 );

        if( res * y > 0 )
        {
            success += 1.0;
        }
    }

    std::cout << "success rate : " << success << "%\n";
    /*
    LinearModel lm(2);
    for( int i = 0; i < 10; i++ )
        lm.addElement( 3, 0.5, 0.5, 1.0 );
    lm.printElements();
    lm.train( 10000, 0.001 );
    */
    /*
    ExempleRosenblatt er( 200, 3, -1.0, 0.7, 0.0001 );

    // INITIALIZATION
    er.populate_points_random();
    er.init_classify();

    // MODEL CREATION
    er.generate_weights();

    // MODEL TRAINING USING ROSENBLATT'S RULE
    er.train(100000, 1000);

    // PRINT FUNCTIONS
    er.print_points("points.txt");
    er.print_background("bg.txt");
    er.print_MSE("MSE.txt");
    */
}