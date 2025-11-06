#include "MLP.hpp"

////////////////////////////////////////////////////////////////////////////////////////////// INITIALISATION

void MLP::init_matrices()
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
    _delta = new float[nb_neurons];

    for( int i = 0; i < nb_weights; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        _W[i] = weight * 2.0 - 1.0;
    }

    for( int i = 0; i < nb_neurons; i++ )
    {
        _X[i] = 0;
        _delta[i] = 0;
    }

    for( int l = 0; l < L; l++ )
        *X(l, 0) = 1.0;
}

////////////////////////////////////////////////////////////////////////////////////////////// ACCESS METHODS

float* MLP::W( int layer, int neuron_out, int neuron_in )
{
    if( layer <= 0 || layer > L || neuron_out < 0 || neuron_out > d[layer-1] || neuron_in <= 0 || neuron_in > d[layer] )
    {
        return NULL;
    }

    int offset = 0;
    for( int l = 1; l < layer; l++ )    
        offset += (d[l-1]+1)*d[l];

    return &( _W[offset + neuron_out*d[layer] + neuron_in - 1] );
}

float* MLP::getNeuronsData( float* array, int layer, int neuron )
{
    if( layer < 0 || layer >= L || neuron < 0 || neuron > d[layer] )
    {
        return NULL;
    }

    int offset = 0;
    for( int l = 0; l < layer; l++ )    
        offset += d[l] + 1;

    return &( array[offset + neuron] );
}

float* MLP::X( int layer, int neuron )
{
    return getNeuronsData( _X, layer, neuron );
}

float* MLP::delta( int layer, int neuron )
{
    return getNeuronsData( _delta, layer, neuron );
}

float* MLP::inputs( int elemIndex, int componentIndex )
{
    if( elemIndex < 0 || elemIndex >= nb_elements || componentIndex < 0 || componentIndex >= d[0] )
    {
        return NULL;
    }

    return &(_inputs[elemIndex*d[0]+componentIndex]);
}

float* MLP::expected_outputs( int elemIndex, int componentIndex  )
{
    if( elemIndex < 0 || elemIndex >= nb_elements || componentIndex < 0 || componentIndex >= d[L-1] )
    {
        return NULL;
    }
    
    return &(_expected_outputs[elemIndex*d[L-1]+componentIndex]);
}

////////////////////////////////////////////////////////////////////////////////////////////// ELEMENTS INITIALISATION

void MLP::initElements( int nb_elements_alloc )
{
    if( _inputs ) delete[] _inputs;
    if( _expected_outputs ) delete[] _expected_outputs;

    _inputs = new float[nb_elements_alloc*d[0]];
    _expected_outputs = new float[nb_elements_alloc*d[L-1]];
}

void MLP::addElement( int count, ... )
{
    if( count != d[0]+d[L-1] )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << d[0]+d[L-1] << " needed." << std::endl;
        return; 
    }

    va_list args;
    va_start( args, count );

    nb_elements++;

    for( int i = 0; i < d[0]; i++ )
        *inputs( nb_elements-1, i ) = va_arg( args, double );

    for( int i = 0; i < d[L-1]; i++ )
        *expected_outputs( nb_elements-1, i ) = va_arg( args, double );

    va_end( args ); 
}

void MLP::addElementArray( int count, float* array )
{
    if( count != d[0]+d[L-1] )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << d[0]+d[L-1] << " needed." << std::endl;
        return; 
    }

    nb_elements++;
    int index = 0;

    for( int i = 0; i < d[0]; i++, index++ )
        *inputs( nb_elements-1, i ) = array[index];

    for( int i = 0; i < d[L-1]; i++, index++ )
        *expected_outputs( nb_elements-1, i ) = array[index];
}

void MLP::printElements()
{
    /*
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
*/
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

void MLP::propagate( int k, bool is_used_for_classification )
{   
    // Initialisation de la couche d'entrée
    for( int j = 0; j < d[0]; j++ )
        *X(0, j+1) = *inputs(k, j);

    // Calcul de chaque neurone 
    for( int l = 1; l < L; l++ )
        for( int j = 1; j <= d[l]; j++ )
        {
            float signal = 0;

            for( int i = 0; i <= d[l-1]; i++ )
                signal += *W(l, i, j) * *X(l-1, i);

            if( is_used_for_classification || l != L-1 )
                signal = tanh(signal);
            
            *X(l, j) = signal;
        }
}

void MLP::retropropagate( int k, bool is_used_for_classification, float alpha )
{
    // Initialisation du delta de la couche de sortie
    for( int j = 1; j <= d[L-1]; j++ )
    {
        *delta(L-1, j) = *X(L-1, j) - *expected_outputs(k, j-1);

        if( is_used_for_classification )
            *delta(L-1, j) *= ( 1.0 - pow( *X(L-1, j), 2.0 ) );
    }

    // Calcul du reste des delta
    for( int l = L-1; l >= 2; l-- )
        for( int i = 1; i <= d[l-1]; i++ )
        {
            float total = 0;

            for( int j = 1; j <= d[l]; j++ )
                total += *W(l, i, j) * *delta(l, j);

           *delta(l-1,i) = ( 1.0 - pow(*X(l-1, i), 2.0) ) * total;
        }
        
    // Correction des poids  
    for( int l = 1; l < L; l++ )
        for( int i = 0; i <= d[l-1]; i++ )
            for( int j = 1; j <= d[l]; j++ )
                *W(l, i, j) -= alpha * *X(l-1, i) * *delta(l, j);
}

void MLP::train( int nb_iterations, float alpha, bool is_used_for_classification, int MSE_interval )
{
    int MSE_index = 0;
    float tmpMSE = 0.0;
    if( MSE_interval > 0 )
    {
        if( !_MSE )
            delete[] _MSE;
        
        nb_MSE = nb_iterations/MSE_interval+1;
        _MSE = new float[nb_MSE];
    }

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_elements;

        propagate( k, is_used_for_classification );
        retropropagate( k, is_used_for_classification, alpha );

        if( MSE_interval > 0 )
        {
            for( int j = 0; j < d[L-1]; j++ )
                tmpMSE += pow( *X(L-1, j+1) - *expected_outputs(k, j), 2.0 );
            
            if( (i+1)%MSE_interval == 0 )
            {
                _MSE[MSE_index] = tmpMSE / static_cast<float>( d[L-1] * MSE_interval );
                MSE_index++;
                tmpMSE = 0.0;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////// PREDICTING METHODS

void MLP::generatePrediction( bool is_used_for_classification, int count, ... )
{
    if( count != d[0] )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << d[0] << " needed." << std::endl;
        return; 
    }

    va_list args;
    va_start( args, count );

    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = va_arg( args, double );

    va_end( args ); 

    for( int l = 1; l < L; l++ )
        for( int j = 1; j <= d[l]; j++ )
        {
            float signal = 0;

            for( int i = 0; i <= d[l-1]; i++ )
                signal += *W(l, i, j) * *X(l-1, i);

            if( is_used_for_classification || l != L-1 )
                signal = tanh(signal);
            
            *X(l, j) = signal;
        }
}

void MLP::generatePredictionArray( bool is_used_for_classification, int count, float* array )
{
    if( count != d[0] )
    {
        std::cout << "Could not add element : invalid number of arguments. " << count << " given, " << d[0] << " needed." << std::endl;
        return; 
    }

    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = array[i-1];

    for( int l = 1; l < L; l++ )
        for( int j = 1; j <= d[l]; j++ )
        {
            float signal = 0;

            for( int i = 0; i <= d[l-1]; i++ )
                signal += *W(l, i, j) * *X(l-1, i);

            if( is_used_for_classification || l != L-1 )
                signal = tanh(signal);
            
            *X(l, j) = signal;
        }
}

float MLP::getPrediction( int index )
{
    if( index < 0 || index >= d[L-1] )
    {
        std::cout << "invalid neuron index. " << index << " is not between 0 and " << d[L-1]-1 << "." << std::endl;
        return 999999.9;
    }
    
    return *X(L-1, index+1);
}

float MLP::test( bool is_used_for_classification )
{
    float success = 0;

    for( int k = 0; k < nb_elements; k++ )
    {
        propagate( k, is_used_for_classification );

        bool toIncrement = true;
        for( int i = 1; i < d[L-1] && toIncrement; i++ )
            toIncrement = ( ( *X(L-1, i) > 0 &&  *expected_outputs(k, i-1) > 0 ) || ( *X(L-1, i) < 0 &&  *expected_outputs(k, i-1) < 0 ));

        if( toIncrement )   
            success += 1.0;
    }

    return (success*100.0)/static_cast<float>(nb_elements);
}