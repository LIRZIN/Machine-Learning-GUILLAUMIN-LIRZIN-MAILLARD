#include "exemple_Rosenblatt.hpp"

int main()
{
    ExempleRosenblatt er( 200, 3, -1.0, 0.7 );

    // INITIALIZATION
    er.populate_points_random( 0.0f, 1.0f );
    er.generate_weights();
    er.init_classify();

    // MODEL CREATION


    // MODEL TRAINING USING ROSENBLATT'S RULE
}