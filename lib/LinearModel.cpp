#include "LinearModel.hpp"

LinearModel::LinearModel( int init_nb_components_of_element ) 
{
    nb_components_of_element = init_nb_components_of_element;

    weights = Eigen::VectorXf( nb_components_of_element+1 );

    for( int i = 0; i <= nb_components_of_element; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights[i] = weight * 2.0 - 1.0;
    }
}

void LinearModel::initElements( int nbElements )
{
    input = Eigen::MatrixXf( nbElements, nb_components_of_element+1 );
    output = Eigen::VectorXf( nbElements );
    nb_elements = 0;
}

void LinearModel::addElement( int count, ... )
{
    va_list args;
    va_start( args, count );

    input(nb_elements, 0) = 1; 
    
    for( int i = 1; i <= nb_components_of_element; i++ )
        input(nb_elements, i) = va_arg( args, double );

    output[nb_elements] = va_arg( args, double );

    nb_elements++;

    va_end( args ); 
}

void LinearModel::addElementArray( float* array )
{
    input(nb_elements, 0) = 1; 

    for( int i = 0; i < nb_components_of_element; i++ )
        input(nb_elements, i+1) = array[i];

    output[nb_elements] = array[nb_components_of_element];

    nb_elements++;
}

void LinearModel::print()
{
    std::cout << "X = \n" << input << std::endl << std::endl;
    std::cout << "Y^T = \n" << output.transpose() << std::endl << std::endl;
    std::cout << "W^T = \n" << weights.transpose() << std::endl << std::endl;
}

float LinearModel::predict( int count, ... )
{
    va_list args;
    va_start( args, count );

    Eigen::VectorXf X_k_with_one( nb_components_of_element + 1 );
    X_k_with_one[0] = 1;
    
    for( int i = 1; i <= nb_components_of_element; i++ )
        X_k_with_one[i] = va_arg( args, double );

    va_end( args ); 
    return predictVector(X_k_with_one);
}

float LinearModel::predictArray( float* array )
{
    Eigen::VectorXf X_k_with_one( nb_components_of_element + 1 );
    X_k_with_one[0] = 1;

    for( int i = 0; i < nb_components_of_element; i++ )
        X_k_with_one[i+1] = array[i];

    return predictVector(X_k_with_one);
}

float LinearModel::predictVector( Eigen::VectorXf& X_k_with_one )
{
    Eigen::MatrixXf result = weights.transpose() * X_k_with_one;
    //return is_used_for_classification ? ( result(0,0) < 0 ? -1.0 : 1.0 ) : result(0,0);
    return is_used_for_classification ? tanh(result(0,0)) : result(0,0);
}

void LinearModel::train( int nb_iterations, float alpha, int MSE_interval )
{
    int MSE_index = 0;
    float tmpMSE = 0.0;
    if( MSE_interval > 0 )
        _MSE = Eigen::VectorXf( nb_iterations/MSE_interval );

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        Eigen::VectorXf X_k_with_one = input.row(k);
        float Y_k = output[k];

        float g_X_k = predictVector(X_k_with_one);
        weights = weights + alpha * ( Y_k - g_X_k ) * X_k_with_one;

        if( MSE_interval > 0 )
        {
            tmpMSE += ( Y_k - g_X_k );
            
            if( (i+1)%MSE_interval == 0 )
            {
                _MSE[MSE_index] = tmpMSE / static_cast<float>( MSE_interval );
                MSE_index++;
                tmpMSE = 0.0;
            }
        }
    }
}

void LinearModel::quickTrain()
{
    weights = ( ( input.transpose() * input ).inverse() * input.transpose() ) * output;
}

float LinearModel::test()
{
    float success = 0;

    for( int k = 0; k < nb_elements; k++ )
    {
        Eigen::VectorXf X_k_with_one = input.row(k);
        double Y_k = output[k];

        float g_X_k = predictVector(X_k_with_one);

        if( ( Y_k < 0 && g_X_k < 0 ) || ( Y_k > 0 && g_X_k > 0 ) )
            success += 1.0;
    }

    return (success*100.0)/static_cast<float>(nb_elements);
}