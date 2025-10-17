#include "LinearModel.hpp"

LinearModel::LinearModel( int init_nb_components_of_element ) 
{
    nb_components_of_element = init_nb_components_of_element;

    input = Eigen::MatrixXd( nb_components_of_element+1, 1 );
    output = Eigen::VectorXd( 1 );
    weights = Eigen::VectorXd( nb_components_of_element+1 );

    for( int i = 0; i <= nb_components_of_element; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights[i] = weight * 2.0 - 1.0;
    }
}

void LinearModel::addElement( int count, ... )
{
    static bool first_time = true;

    if( count != nb_components_of_element+1 )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << nb_components_of_element+1 << " needed." << std::endl;
        return; 
    }

    va_list args;
    va_start( args, count );
    Eigen::VectorXd new_input( nb_components_of_element+1 );
    double new_output;
    
    new_input[0] = 1;

    for( int i = 1; i < nb_components_of_element+1; i++ )
    {
        new_input[i] = va_arg( args, double );
    }

    new_output = va_arg( args, double );

    if( first_time )
    {
        input.col( 0 ) = new_input;
        output[0] = new_output;
        nb_elements = 1;
        first_time = false;
    }
    else 
    {
        input.conservativeResize( input.rows(), input.cols()+1 );
        input.col( input.cols()-1 ) = new_input;

        output.conservativeResize( output.size()+1 );
        output[output.size()-1] = new_output;

        nb_elements++;
    }

    va_end( args ); 
}

void LinearModel::printElements()
{
    std::cout << "X = \n" << input << std::endl << std::endl;
    std::cout << "Y^T = \n" << output.transpose() << std::endl << std::endl;
    std::cout << "W^T = \n" << weights.transpose() << std::endl << std::endl;
}

double LinearModel::predict( Eigen::VectorXd& X_k_with_one )
{
    Eigen::MatrixXd result = weights.transpose() * X_k_with_one;
    return tanh( result(0,0) );//result(0,0) < 0 ? -1.0 : 1.0;
}

void LinearModel::train( int nb_iterations, double alpha )
{
    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        Eigen::VectorXd X_k_with_one = input.col(k);
        double Y_k = output[k];

        float g_X_k = predict(X_k_with_one);
        weights = weights + alpha * ( Y_k - g_X_k ) * X_k_with_one;
    }
}