#include "LinearModel.hpp"

////////////////////////////////////////////////////////////////////////////////////////////// INITIALISATION

void LinearModel::init_matrices()
{
    // Initialize the RNG
    srand( time(0) );

    int nb_neurons = d[0]+1;
    int nb_weights = 0;

    for( int l = 1; l < L; l++ )
    {
        nb_neurons += d[l] + 1;
        nb_weights += (d[l-1]+1)*d[l];
    }

    _W = new float[nb_weights];
    _X = new float[nb_neurons];

    for( int i = 0; i < nb_weights; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        _W[i] = weight * 2.0 - 1.0;
    }

    for( int i = 0; i < nb_neurons; i++ )
        _X[i] = 0;

    for( int l = 0; l < L; l++ )
        *X(l, 0) = 1.0;
}

////////////////////////////////////////////////////////////////////////////////////////////// ACCESS METHODS

float* LinearModel::W( int layer, int neuron_out, int neuron_in )
{
    if( layer <= 0 || layer > L )
        throw std::runtime_error(std::string("layer donné n'est pas entre 1 et " + std::to_string(L-1) + " comme il devrait l'être ( " + std::to_string(layer) + " donné à " + __FUNCTION__ + "() )"));
    if( neuron_out < 0 || neuron_out > d[layer-1] )
        throw std::runtime_error(std::string("neuron_out donné n'est pas entre 0 et " + std::to_string(d[layer-1] ) + " comme il devrait l'être ( " + std::to_string(neuron_out) + " donné à " + __FUNCTION__ + "() )"));
    if( neuron_in <= 0 || neuron_in > d[layer] )
        throw std::runtime_error(std::string("neuron_in donné n'est pas entre 1 et " + std::to_string(d[layer]) + " comme il devrait l'être ( " + std::to_string(neuron_in) + " donné à " + __FUNCTION__ + "() )"));

    int offset = 0;
    for( int l = 1; l < layer; l++ )    
        offset += (d[l-1]+1)*d[l];

    return &( _W[offset + neuron_out*d[layer] + neuron_in - 1] );
}

float* LinearModel::X( int layer, int neuron )
{
    if( layer < 0 || layer >= L )
        throw std::runtime_error(std::string("layer donné n'est pas entre 0 et " + std::to_string(L-1) + " comme il devrait l'être ( " + std::to_string(layer) + " donné à " + __FUNCTION__ + "() )"));
    if( neuron < 0 || neuron > d[layer] )
        throw std::runtime_error(std::string("neuron donné n'est pas entre 0 et " + std::to_string(d[layer]) + " comme il devrait l'être ( " + std::to_string(neuron) + " donné à " + __FUNCTION__ + "() )"));

    int offset = 0;
    for( int l = 0; l < layer; l++ )    
        offset += d[l] + 1;

    return &( _X[offset + neuron] );
}

float* LinearModel::inputs( int elemIndex, int componentIndex )
{
    if( elemIndex < 0 || elemIndex >= nb_elements )
        throw std::runtime_error(std::string("elemIndex donné n'est pas entre 0 et " + std::to_string(nb_elements-1) + " comme il devrait l'être ( " + std::to_string(elemIndex) + " donné à " + __FUNCTION__ + "() )"));
    if( componentIndex < 0 || componentIndex >= d[0] )
        throw std::runtime_error(std::string("componentIndex donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(componentIndex) + " donné à " + __FUNCTION__ + "() )")); 

    return &(_inputs[elemIndex*d[0]+componentIndex]);
}

float* LinearModel::expected_outputs( int elemIndex, int componentIndex  )
{
    if( elemIndex < 0 || elemIndex >= nb_elements )
        throw std::runtime_error(std::string("elemIndex donné n'est pas entre 0 et " + std::to_string(nb_elements-1) + " comme il devrait l'être ( " + std::to_string(elemIndex) + " donné à " + __FUNCTION__ + "() )"));
    if( componentIndex < 0 || componentIndex >= d[L-1] )
        throw std::runtime_error(std::string("componentIndex donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(componentIndex) + " donné à " + __FUNCTION__ + "() )"));
    
    return &(_expected_outputs[elemIndex*d[L-1]+componentIndex]);
}

////////////////////////////////////////////////////////////////////////////////////////////// ELEMENTS INITIALISATION

void LinearModel::initElements( int nb_elements_alloc )
{
    if( _inputs ) delete[] _inputs;
    if( _expected_outputs ) delete[] _expected_outputs;

    _inputs = new float[nb_elements_alloc*d[0]];
    _expected_outputs = new float[nb_elements_alloc*d[L-1]];
}

void LinearModel::addElement( ... )
{
    va_list args;
    va_start( args, d[0]+d[L-1] );

    nb_elements++;

    for( int i = 0; i < d[0]; i++ )
        *inputs( nb_elements-1, i ) = va_arg( args, double );

    for( int i = 0; i < d[L-1]; i++ )
        *expected_outputs( nb_elements-1, i ) = va_arg( args, double );

    va_end( args ); 
}

void LinearModel::addElementArray( float* array )
{
    nb_elements++;
    int index = 0;

    for( int i = 0; i < d[0]; i++, index++ )
        *inputs( nb_elements-1, i ) = array[index];

    for( int i = 0; i < d[L-1]; i++, index++ )
        *expected_outputs( nb_elements-1, i ) = array[index];
}

void LinearModel::print()
{
    std::cout << "d = ";
    for( int i = 0; i < L; i++ )
        std::cout << d[i] << " ";
    std::cout << std::endl;

    std::cout << "inputs = " << std::endl;
    for( int i = 0; i < nb_elements; i++ )
    {
        for( int j = 0; j < d[0]; j++ )
            std::cout << *inputs(i, j) << " ";

        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "expected_outputs = " << std::endl;
    for( int i = 0; i < nb_elements; i++ )
    {
        for( int j = 0; j < d[L-1]; j++ )
            std::cout << *expected_outputs(i, j) << " ";

        std::cout << std::endl;
    }
    std::cout << std::endl;

    for( int l = 1; l < L; l++ )
    {
        std::cout << "weights[" << l-1 << "] = " << std::endl;
        for( int i = 1; i <= d[l]; i++ )
        {
            for( int j = 0; j <= d[l-1]; j++ ) 
                std::cout << *W(l, j, i) << " ";

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    int max_neurons_layer  = d[0];
    for( int l = 1; l < L; l++ )
        if( max_neurons_layer < d[l] )
            max_neurons_layer = d[l];

    std::cout << "X = " << std::endl;
    for( int n = 0; n <= max_neurons_layer; n++ )
    {
        for( int l = 0; l < L; l++ )
            if( n <= d[l] )
                std::cout << *X(l, n) << " ";
            else 
                std::cout << "  ";

        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "delta = " << std::endl;
    for( int n = 0; n <= max_neurons_layer; n++ )
    {
        for( int l = 0; l < L; l++ )
            if( n <= d[l] )
                std::cout << *delta(l, n) << " ";
            else 
                std::cout << "  ";

        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////// TRAINING METHODS

double LinearModel::predict( Eigen::VectorXd& X_k_with_one )
{
    Eigen::MatrixXd result = weights.transpose() * X_k_with_one;
    return tanh( result(0,0) );//result(0,0) < 0 ? -1.0 : 1.0;
}

void LinearModel::train( int nb_iterations, float alpha )
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

void LinearModel::train( int nb_iterations, float alpha, int MSE_interval )
{
    int MSE_index = 0;
    float tmpMSE = 0.0;
    if( MSE_interval > 0 )
    {
        if( !_MSE )
            delete[] _MSE;
        
        nb_MSE = nb_iterations/MSE_interval;
        _MSE = new float[nb_MSE];
    }

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        propagate( k );
        retropropagate( k, alpha );

        if( MSE_interval > 0 )
        {
            float sampleMSE = 0.0;
            for( int j = 0; j < d[L-1]; j++ )
            {
                float diff = *X(L-1, j+1) - *expected_outputs(k, j);
                sampleMSE += diff * diff;
            }
            tmpMSE += sampleMSE/static_cast<float>( d[L-1] );
            
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
    if( L != 2 )
        throw std::runtime_error(std::string("On ne peut pas déterminer les poids optimaux en utilisant la formule normale quand le réseau a des couches cachées ( dans " + std::string(__FUNCTION__) + "() )"));

    Eigen::MatrixXd eigenInputs( nb_elements, d[0]+1 );
    Eigen::MatrixXd eigenExpected_outputs( nb_elements, d[L-1] );

    for( int i = 0; i < nb_elements; i++ )
    {
        eigenInputs(i, 0) = 1.0;
        for( int j = 0; j < d[0]; j++ )
            eigenInputs(i, j+1) = *inputs(i, j);

        for( int j = 0; j < d[L-1]; j++ )
            eigenExpected_outputs(i, j) = *expected_outputs(i, j);
    }

    Eigen::MatrixXd weights = ( ( eigenInputs.transpose() * eigenInputs ).inverse() * eigenInputs.transpose() ) * eigenExpected_outputs;

    for( int i = 0; i <= d[0]; i++ )
        for( int j = 1; j <= d[L-1]; j++ )
            *W(1, i, j) = weights(i, j-1);
}

////////////////////////////////////////////////////////////////////////////////////////////// PREDICTING METHODS

void LinearModel::generatePrediction( ... )
{
    va_list args;
    va_start( args, d[0] );

    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = va_arg( args, double );

    va_end( args ); 

    propagate( -1 );
}

void LinearModel::generatePredictionArray( float* array )
{
    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = array[i-1];

    propagate( -1 );
}

float LinearModel::getPrediction( int index )
{
    if( index < 0 || index >= d[L-1] )
        throw std::runtime_error(std::string("index donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(index) + " donné à " + __FUNCTION__ + "() )"));

    return -1;//*X(L-1, index+1);
}

float LinearModel::test()
{
    float success = 0;

    for( int k = 0; k < nb_elements; k++ )
    {
        propagate( k );

        bool toIncrement = true;
        for( int i = 1; i <= d[L-1] && toIncrement; i++ )
            toIncrement = ( ( *X(L-1, i) > 0 &&  *expected_outputs(k, i-1) > 0 ) || ( *X(L-1, i) < 0 &&  *expected_outputs(k, i-1) < 0 ));

        if( toIncrement )   
            success += 1.0;
    }

    return (success*100.0)/static_cast<float>(nb_elements);
}