#include "MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::init_matrices()
{
    // Initialize the RNG
    srand( time(0) );

    input = Eigen::MatrixXd( nb_neurons_in_input_layer+1, 1 );
    output = Eigen::MatrixXd( nb_neurons_in_output_layer, 1 );
    predictedOutput = Eigen::VectorXd( nb_neurons_in_output_layer );
    weights = new Eigen::MatrixXd[nb_layers-1];

    for( int i = 0; i < nb_layers-1; i++ )
    {
        weights[i] = Eigen::MatrixXd( nb_neurons_in_layer[i]+1, nb_neurons_in_layer[i+1] );

        for( int j = 0; j < nb_neurons_in_layer[i]+1; j++ )
        {
            for( int k = 0; k < nb_neurons_in_layer[i+1]; k++ )
            {
                float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                weights[i](j, k) = weight * 2.0 - 1.0;
            }
        }
    }
}

Eigen::VectorXd MultiLayerPerceptron::getInWeights( int layer, int neuron )
{
    if( layer < 1 || layer > nb_layers )
    {
        std::cout << "invalid layer index. " << layer << " is not between 1 and " << nb_layers << "." << std::endl;
        return Eigen::Vector2d();
    }
    if( neuron < 0 || neuron >= nb_neurons_in_layer[layer] )
    {
        std::cout << "invalid neuron index. " << neuron << " is not between 0 and " << nb_neurons_in_layer[layer]-1 << "." << std::endl;
        return Eigen::Vector2d();
    }

    return weights[layer-1].col( neuron );
}

void MultiLayerPerceptron::setInWeights( int layer, int neuron, Eigen::VectorXd& weight )
{
    if( layer < 1 || layer > nb_layers )
    {
        std::cout << "invalid layer index. " << layer << " is not between 1 and " << nb_layers << "." << std::endl;
        return;
    }
    if( neuron < 0 || neuron >= nb_neurons_in_layer[layer] )
    {
        std::cout << "invalid neuron index. " << neuron << " is not between 0 and " << nb_neurons_in_layer[layer]-1 << "." << std::endl;
        return;
    }

    weights[layer-1].col( neuron ) = weight;
}

Eigen::VectorXd MultiLayerPerceptron::getOutWeights( int layer, int neuron )
{
    if( layer < 0 || layer > nb_layers - 2 )
    {
        std::cout << "invalid layer index. " << layer << " is not between 0 and " << nb_layers - 2 << "." << std::endl;
        return Eigen::Vector2d();
    }
    if( neuron < 0 || neuron >= nb_neurons_in_layer[layer]+1 )
    {
        std::cout << "invalid neuron index. " << neuron << " is not between 0 and " << nb_neurons_in_layer[layer] << "." << std::endl;
        return Eigen::Vector2d();
    }

    return weights[layer].row( neuron );
}

void MultiLayerPerceptron::setOutWeights( int layer, int neuron, Eigen::VectorXd& weight )
{
    if( layer < 0 || layer > nb_layers - 2 )
    {
        std::cout << "invalid layer index. " << layer << " is not between 0 and " << nb_layers - 2 << "." << std::endl;
        return;
    }
    if( neuron < 0 || neuron >= nb_neurons_in_layer[layer]+1 )
    {
        std::cout << "invalid neuron index. " << neuron << " is not between 0 and " << nb_neurons_in_layer[layer] << "." << std::endl;
        return;
    }

    weights[layer].row( neuron ) = weight;
}

double MultiLayerPerceptron::activation_function( bool is_used_for_classification, double value )
{
    return (is_used_for_classification)?tanh( value ):value;
}

void MultiLayerPerceptron::compute_neuron_values( Eigen::VectorXd* neurons, bool is_used_for_classification )
{
    for( int i = 1; i < nb_layers-1; i++ )
        for( int j = 0; j < nb_neurons_in_layer[i]; j++ )
        {
            neurons[i][j+1] = tanh( getInWeights( i, j ).transpose() * neurons[i-1] );
        }
    
    for( int i = 0; i < nb_neurons_in_output_layer; i++ )
    {
        neurons[nb_layers-1][i] = activation_function( is_used_for_classification, getInWeights( nb_layers-1, i ).transpose() * neurons[nb_layers-2] );
    }
}

void MultiLayerPerceptron::update_weights( Eigen::VectorXd* neurons, Eigen::VectorXd& output_k, double alpha, bool is_used_for_classification )
{
    Eigen::VectorXd* delta = new Eigen::VectorXd[nb_layers-1];
    for( int i = 1; i < nb_layers; i++ )
        delta[i-1] = Eigen::VectorXd( nb_neurons_in_layer[i] ); 

    delta[nb_layers-2] = output_k - neurons[nb_layers-1];
    if( is_used_for_classification )
        for( int i = 0; i < nb_neurons_in_output_layer; i++ )
            delta[nb_layers-2][i] *= ( 1.0 - pow( neurons[nb_layers-1][i], 2.0 ) );
    
    for( int i = nb_layers-2; i > 0; i-- )
        for( int j = 0; j < nb_neurons_in_layer[i]; j++ )
            delta[i-1][j] = ( 1.0 - pow( neurons[i-1][j+1], 2.0 ) ) * ( getOutWeights( i, j ).transpose() * delta[i] )(0, 0);
    
    for( int i = nb_layers-2; i >= 0; i-- )
        for( int j = 0; j <= nb_neurons_in_layer[i]; j++ )
        {
            Eigen::VectorXd weight = getOutWeights( i, j ) - alpha * neurons[i][j] * delta[i];
            setOutWeights( i, j, weight );
        }

    delete[] delta;
}

void MultiLayerPerceptron::addElement( int count, ... )
{
    static bool first_time = true;

    if( count != nb_neurons_in_input_layer+nb_neurons_in_output_layer )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << nb_neurons_in_input_layer+nb_neurons_in_output_layer << " needed." << std::endl;
        return; 
    }

    va_list args;
    va_start( args, count );
    Eigen::VectorXd new_input( nb_neurons_in_input_layer+1 );
    Eigen::VectorXd new_output( nb_neurons_in_output_layer );
    
    new_input[0] = 1;

    for( int i = 0; i < nb_neurons_in_input_layer+1; i++ )
    {
        new_input[i] = va_arg( args, double );
    }
    for( int i = 0; i < nb_neurons_in_output_layer; i++ )
    {
        new_output[i] = va_arg( args, double );
    }

    if( first_time )
    {
        input.col( 0 ) = new_input;
        output.col( 0 ) = new_output;
        first_time = false;
    }
    else 
    {
        input.conservativeResize( input.rows(), input.cols()+1 );
        input.col( input.cols()-1 ) = new_input;

        output.conservativeResize( output.rows(), output.cols()+1 );
        output.col( output.cols()-1 ) = new_output;
    }

    va_end( args ); 
}

void MultiLayerPerceptron::addElementArray( int count, double* array )
{
    static bool first_time = true;

    if( count != nb_neurons_in_input_layer+nb_neurons_in_output_layer )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << nb_neurons_in_input_layer+nb_neurons_in_output_layer << " needed." << std::endl;
        return; 
    }

    Eigen::VectorXd new_input( nb_neurons_in_input_layer+1 );
    Eigen::VectorXd new_output( nb_neurons_in_output_layer );
    
    new_input[0] = 1;
    int index = 0;

    for( int i = 1; i < nb_neurons_in_input_layer+1; i++ )
    {
        new_input[i] = array[index++];
    }
    for( int i = 0; i < nb_neurons_in_output_layer; i++ )
    {
        new_output[i] = array[index++];
    }

    if( first_time )
    {
        input.col( 0 ) = new_input;
        output.col( 0 ) = new_output;
        first_time = false;
    }
    else 
    {
        input.conservativeResize( input.rows(), input.cols()+1 );
        input.col( input.cols()-1 ) = new_input;

        output.conservativeResize( output.rows(), output.cols()+1 );
        output.col( output.cols()-1 ) = new_output;
    }
}

void MultiLayerPerceptron::printElements()
{
    std::cout << "X = \n" << input << std::endl << std::endl;
    std::cout << "Y = \n" << output << std::endl << std::endl;
    for( int i = 0; i < nb_layers; i++ )
        std::cout << "W[" << i << "] = \n" << weights[i] << std::endl << std::endl;
}

void MultiLayerPerceptron::quickTrain()
{
    if( nb_layers != 2 )
    {
        std::cout << "cannot quick train if network has hidden layers" << std::endl;
        return;
    }
    
    weights[0] = ( ( input * input.transpose() ).inverse() * input ) * output.transpose();
}

void MultiLayerPerceptron::train( int nb_iterations, double alpha, bool is_used_for_classification, int MSE_interval )
{
    Eigen::VectorXd* neurons = new Eigen::VectorXd[nb_layers];
    for( int i = 0; i < nb_layers; i++ )
     {
        neurons[i] = Eigen::VectorXd( nb_neurons_in_layer[i] + ((i==(nb_layers-1))?0:1) );
        neurons[i][0] = 1.0;
     }

    int MSE_index = 0;
    double tmpMSE = 0.0;
    if( MSE_interval > 0 )
        MSE_values = Eigen::VectorXd( nb_iterations/MSE_interval );

    int nb_elements = input.cols();

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        neurons[0] = input.col(k);
        Eigen::VectorXd output_k = output.col(k);

        compute_neuron_values( neurons, is_used_for_classification );
        update_weights( neurons, output_k, alpha, is_used_for_classification );

        if( MSE_interval > 0 )
        {
            Eigen::VectorXd delta_output = neurons[nb_layers-1] - output_k;
            for( int j = 0; j < nb_neurons_in_output_layer; j++ )
                tmpMSE += pow( delta_output[j], 2.0 );
            
            if( i%MSE_interval == 0 )
            {
                MSE_values[MSE_index] = tmpMSE / static_cast<double>( nb_neurons_in_output_layer * MSE_interval );
                MSE_index++;
                tmpMSE = 0.0;
            }
        }
    }

    delete[] neurons;
}

void MultiLayerPerceptron::generatePrediction( bool is_used_for_classification, int count, ... )
{
    if( count != nb_neurons_in_input_layer )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << nb_neurons_in_input_layer << " needed." << std::endl;
        return; 
    }

    va_list args;
    va_start( args, count );
    Eigen::VectorXd* neurons = new Eigen::VectorXd[nb_layers];
    for( int i = 0; i < nb_layers; i++ )
    {
        neurons[i] = Eigen::VectorXd( nb_neurons_in_layer[i] + ((i==(nb_layers-1))?0:1));
        neurons[i][0] = 1.0;
    }

    for( int i = 1; i <= nb_neurons_in_input_layer; i++ )
        neurons[0][i] = va_arg( args, double );

    va_end( args ); 

    compute_neuron_values( neurons, is_used_for_classification );

    predictedOutput = neurons[nb_layers-1];
    
    delete[] neurons;
}

void MultiLayerPerceptron::generatePredictionArray( bool is_used_for_classification, int count, double* array )
{
    if( count != nb_neurons_in_input_layer )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << nb_neurons_in_input_layer << " needed." << std::endl;
        return; 
    }
    
    Eigen::VectorXd* neurons = new Eigen::VectorXd[nb_layers];
    for( int i = 0; i < nb_layers; i++ )
        neurons[i] = Eigen::VectorXd( nb_neurons_in_layer[i] + 1 );
    
    neurons[0][0] = 1;

    for( int i = 1; i < nb_neurons_in_input_layer+1; i++ )
        neurons[0][i] = array[i-1];

    compute_neuron_values( neurons, is_used_for_classification );

    predictedOutput = neurons[nb_layers-1];

    delete[] neurons;
}

double MultiLayerPerceptron::getPrediction( int index )
{
    if( index < 0 || index >= nb_neurons_in_output_layer )
    {
        std::cout << "invalid neuron index. " << index << " is not between 0 and " << nb_neurons_in_output_layer-1 << "." << std::endl;
        return 999999.9;
    }
    
    return predictedOutput[index];
}