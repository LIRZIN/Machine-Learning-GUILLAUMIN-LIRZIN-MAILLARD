//#include "exemple_Rosenblatt.hpp"
//#include "LinearModel.hpp"
#include "MultiLevelPerceptron.hpp"

int main()
{
    srand( time(0) );

    MultiLevelPerceptron mlp(2, 1);
    for( int i = 0; i < 10; i++ )
        mlp.addElement( 3, 0.5, 0.5, 1.0 );
    mlp.printElements();
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