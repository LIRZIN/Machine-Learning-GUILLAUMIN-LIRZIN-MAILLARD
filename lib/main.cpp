#include "exemple_Rosenblatt.hpp"

int main()
{
    srand( time(0) );
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
}