#include "MultiLevelPerceptron.hpp"

void MultiLevelPerceptron::init_matrices()
{
    input = Eigen::MatrixXd( nb_neurons_in_input_layer+1, 1 );
    output = Eigen::VectorXd( 1 );

    if( nb_hidden_layers == 0 )
    {
        input_weights = Eigen::MatrixXd( nb_neurons_in_input_layer+1, nb_neurons_in_output_layer );
    }
    else 
    {
        input_weights = Eigen::MatrixXd( nb_neurons_in_input_layer+1, nb_neurons_in_hidden_layer );
        hidden_weights = Eigen::MatrixXd( nb_neurons_in_hidden_layer+1, nb_neurons_in_hidden_layer*(nb_hidden_layers-1) + nb_neurons_in_output_layer );

        for( int i = 0; i < hidden_weights.rows(); i++ )
        {
            for( int j = 0; j < hidden_weights.cols(); j++ )
            {
                float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                hidden_weights(i, j) = weight * 2.0 - 1.0;
            }
        }
    }
    
    for( int i = 0; i < input_weights.rows(); i++ )
    {
        for( int j = 0; j < input_weights.cols(); j++ )
        {
            float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            input_weights(i, j) = weight * 2.0 - 1.0;
        }
    }
}

void MultiLevelPerceptron::addElement( int count, ... )
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

    for( int i = 1; i < nb_neurons_in_input_layer+1; i++ )
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
        nb_elements = 1;
        first_time = false;
    }
    else 
    {
        input.conservativeResize( input.rows(), input.cols()+1 );
        input.col( input.cols()-1 ) = new_input;

        output.conservativeResize( output.rows(), output.cols()+1 );
        output.col( output.cols()-1 ) = new_output;

        nb_elements++;
    }

    va_end( args ); 
}

void MultiLevelPerceptron::printElements()
{
    std::cout << "X = \n" << input << std::endl << std::endl;
    std::cout << "Y = \n" << output << std::endl << std::endl;
    std::cout << "W[input] = \n" << input_weights << std::endl << std::endl;
    std::cout << "W[hidden] = \n" << hidden_weights << std::endl << std::endl;
}

Eigen::VectorXd MultiLevelPerceptron::getInWeights( int layer, int neuron )
{
    if( layer < 1 && layer > nb_hidden_layers + 2 )
    {
        std::cout << "invalid layer index. " << layer << " is not between 0 and " << nb_hidden_layers+1 << "." << std::endl;
        return Eigen::Vector2d();
    }

    if( layer == 1 )
    {
        if( neuron < 0 && neuron > input_weights.cols() )
        {
            std::cout << "invalid neuron index for layer " << layer << ". " << neuron << " is not between 0 and " << input_weights.cols()-1 << "." << std::endl;
            return Eigen::Vector2d();
        }

        return input_weights.col( neuron );
    }
    if( layer == nb_hidden_layers + 2 )
    {
        if( neuron < 0 && neuron > nb_neurons_in_output_layer )
        {
            std::cout << "invalid neuron index for the last layer. " << neuron << " is not between 0 and " << nb_neurons_in_output_layer-1 << "." << std::endl;
            return Eigen::Vector2d();
        }

        return hidden_weights.col( hidden_weights.cols() - nb_neurons_in_output_layer + neuron );
    }

    if( neuron < 0 && neuron > nb_neurons_in_hidden_layer )
    {
        std::cout << "invalid neuron index for the hidden layer" << layer << ". " << neuron << " is not between 0 and " << nb_neurons_in_hidden_layer-1 << "." << std::endl;
        return Eigen::Vector2d();
    }

    return hidden_weights.col( (layer-2)*nb_neurons_in_hidden_layer + neuron );
}

Eigen::VectorXd MultiLevelPerceptron::getOutWeights( int layer, int neuron )
{
    if( layer < 0 && layer > nb_hidden_layers + 1 )
    {
        std::cout << "invalid layer index. " << layer << " is not between 0 and " << nb_hidden_layers << "." << std::endl;
        return Eigen::Vector2d();
    }

    if( layer == 0 )
    {
        if( neuron < 0 && neuron > input.rows()-2 )
        {
            std::cout << "invalid neuron index for layer " << layer << ". " << neuron << " is not between 0 and " << input.rows()-2 << "." << std::endl;
            return Eigen::Vector2d();
        }

        return input_weights.row( neuron+1 ).transpose();
    }
    if( layer == nb_hidden_layers + 1 )
    {
        if( neuron < 0 && neuron > nb_neurons_in_hidden_layer )
        {
            std::cout << "invalid neuron index for the last layer. " << neuron << " is not between 0 and " << nb_neurons_in_hidden_layer-1 << "." << std::endl;
            return Eigen::Vector2d();
        }

        return hidden_weights.block( neuron+1, hidden_weights.cols() - nb_neurons_in_output_layer, 1, nb_neurons_in_output_layer ).transpose();
    }

    if( neuron < 0 && neuron > nb_neurons_in_hidden_layer )
    {
        std::cout << "invalid neuron index for the hidden layer" << layer << ". " << neuron << " is not between 0 and " << nb_neurons_in_hidden_layer-1 << "." << std::endl;
        return Eigen::Vector2d();
    }

    return hidden_weights.block( neuron+1, layer * nb_neurons_in_hidden_layer, 1, nb_neurons_in_hidden_layer ).transpose();
}

void MultiLevelPerceptron::setOutWeights( int layer, int neuron, Eigen::VectorXd& weights )
{
    if( layer < 0 && layer > nb_hidden_layers + 1 )
    {
        std::cout << "invalid layer index. " << layer << " is not between 0 and " << nb_hidden_layers << "." << std::endl;
        return;
    }

    if( layer == 0 )
    {
        if( neuron < 0 && neuron > input.rows()-2 )
        {
            std::cout << "invalid neuron index for layer " << layer << ". " << neuron << " is not between 0 and " << input.rows()-2 << "." << std::endl;
            return;
        }

        input_weights.row( neuron+1 ) = weights;
    }
    if( layer == nb_hidden_layers + 1 )
    {
        if( neuron < 0 && neuron > nb_neurons_in_hidden_layer )
        {
            std::cout << "invalid neuron index for the last layer. " << neuron << " is not between 0 and " << nb_neurons_in_hidden_layer-1 << "." << std::endl;
            return;
        }

        // hidden_weights.block( neuron+1, hidden_weights.cols() - nb_neurons_in_output_layer, 1, nb_neurons_in_output_layer ).transpose();
    }

    if( neuron < 0 && neuron > nb_neurons_in_hidden_layer )
    {
        std::cout << "invalid neuron index for the hidden layer" << layer << ". " << neuron << " is not between 0 and " << nb_neurons_in_hidden_layer-1 << "." << std::endl;
        return;
    }

    // hidden_weights.block( neuron+1, layer * nb_neurons_in_hidden_layer, 1, nb_neurons_in_hidden_layer ).transpose();
}

void MultiLevelPerceptron::compute_neuron_values( Eigen::VectorXd& input_k, Eigen::MatrixXd& computed_hidden_neurons, Eigen::VectorXd& computed_output_neurons )
{
    if( nb_hidden_layers == 0 )
    {
        for( int i = 0; i < nb_neurons_in_output_layer; i++ )
        {
            computed_output_neurons[i] = tanh( getInWeights( 1, i ).transpose() * input_k );
        }
    }
    else 
    {
        for( int i = 0; i < nb_neurons_in_hidden_layer; i++ )
        {
            computed_hidden_neurons(i+1, 0) = tanh( getInWeights( 1, i ).transpose() * input_k );
        }

        for( int i = 1; i < nb_hidden_layers; i++ )
        {
            for( int j = 0; j < nb_neurons_in_hidden_layer; j++ )
            {
                computed_hidden_neurons(j+1, i) = tanh( getInWeights( i+1, j ).transpose() * computed_hidden_neurons.col(i-1) );
            }
        }

        for( int i = 0; i < nb_neurons_in_output_layer; i++ )
        {
            computed_output_neurons[i] = tanh( getInWeights( nb_hidden_layers+1, i ).transpose() * computed_hidden_neurons.col(nb_hidden_layers-1) );
        }
    }
}

void MultiLevelPerceptron::update_weights( Eigen::VectorXd& input_k, Eigen::VectorXd& output_k, Eigen::MatrixXd& computed_hidden_neurons, Eigen::VectorXd& computed_output_neurons, double alpha )
{
    Eigen::VectorXd output_delta = computed_output_neurons - output_k;

    if( nb_hidden_layers == 0 )
    {
        for( int i = 0; i < nb_neurons_in_input_layer; i++ )
        {
            Eigen::VectorXd new_weight = getOutWeights( 0, i ) - alpha * input_k[i+1] * output_delta;
            setOutWeights( 0, i, new_weight );
        }
    }
    else 
    {
        // double delta = ( 1.0 - pow( input_k[i], 2.0 ) ) * ( getOutWeights( 0, i ).transpose() * output_delta );
    }
}

void MultiLevelPerceptron::train( int nb_iterations, double alpha )
{
    Eigen::MatrixXd computed_hidden_neurons;
    if( nb_hidden_layers > 0 )
    {
        computed_hidden_neurons = Eigen::MatrixXd( nb_neurons_in_hidden_layer+1, nb_hidden_layers );

        for( int i = 0; i < nb_hidden_layers; i++ )
        {
            computed_hidden_neurons(0, i) = 1.0;
        }
    }
    Eigen::VectorXd computed_output_neurons( nb_neurons_in_output_layer );

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        Eigen::VectorXd input_k = input.col(k);
        Eigen::VectorXd ouput_k = output.col(k);

        compute_neuron_values( input_k, computed_hidden_neurons, computed_output_neurons );
        update_weights( input_k, ouput_k, computed_hidden_neurons, computed_output_neurons, alpha );
    }
}

double MultiLevelPerceptron::predict( double x1, double x2 )
{
    Eigen::MatrixXd computed_hidden_neurons;
    Eigen::VectorXd computed_output_neurons( nb_neurons_in_output_layer );

    Eigen::VectorXd input_k(3);
    input_k[0] = 1;
    input_k[1] = x1;
    input_k[2] = x2;

    compute_neuron_values( input_k, computed_hidden_neurons, computed_output_neurons );
    return computed_output_neurons(0, 0);
}