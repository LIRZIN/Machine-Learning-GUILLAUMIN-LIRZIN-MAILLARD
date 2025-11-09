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

float* MLP::getNeuronsData( float* array, int layer, int neuron )
{
    if( layer < 0 || layer >= L )
        throw std::runtime_error(std::string("layer donné n'est pas entre 0 et " + std::to_string(L-1) + " comme il devrait l'être ( " + std::to_string(layer) + " donné à " + __FUNCTION__ + "() )"));
    if( neuron < 0 || neuron > d[layer] )
        throw std::runtime_error(std::string("neuron donné n'est pas entre 0 et " + std::to_string(d[layer]) + " comme il devrait l'être ( " + std::to_string(neuron) + " donné à " + __FUNCTION__ + "() )"));

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
    if( elemIndex < 0 || elemIndex >= nb_elements )
        throw std::runtime_error(std::string("elemIndex donné n'est pas entre 0 et " + std::to_string(nb_elements-1) + " comme il devrait l'être ( " + std::to_string(elemIndex) + " donné à " + __FUNCTION__ + "() )"));
    if( componentIndex < 0 || componentIndex >= d[0] )
        throw std::runtime_error(std::string("componentIndex donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(componentIndex) + " donné à " + __FUNCTION__ + "() )")); 

    return &(_inputs[elemIndex*d[0]+componentIndex]);
}

float* MLP::expected_outputs( int elemIndex, int componentIndex  )
{
    if( elemIndex < 0 || elemIndex >= nb_elements )
        throw std::runtime_error(std::string("elemIndex donné n'est pas entre 0 et " + std::to_string(nb_elements-1) + " comme il devrait l'être ( " + std::to_string(elemIndex) + " donné à " + __FUNCTION__ + "() )"));
    if( componentIndex < 0 || componentIndex >= d[L-1] )
        throw std::runtime_error(std::string("componentIndex donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(componentIndex) + " donné à " + __FUNCTION__ + "() )"));
    
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
        throw std::runtime_error(std::string("count n'est pas égal à " + std::to_string(d[0]+d[L-1]) + " comme il devrait l'être ( " + std::to_string(count) + " donné à " + __FUNCTION__ + "() )"));

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
        throw std::runtime_error(std::string("count n'est pas égal à " + std::to_string(d[0]+d[L-1]) + " comme il devrait l'être ( " + std::to_string(count) + " donné à " + __FUNCTION__ + "() )"));

    nb_elements++;
    int index = 0;

    for( int i = 0; i < d[0]; i++, index++ )
        *inputs( nb_elements-1, i ) = array[index];

    for( int i = 0; i < d[L-1]; i++, index++ )
        *expected_outputs( nb_elements-1, i ) = array[index];
}

void MLP::print()
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

void MLP::propagate( int k )
{   
    // Initialisation de la couche d'entrée
    if( k >= 0 ) // Pour éviter la duplication de code entre l'entrainement et la prédiction
        for( int j = 0; j < d[0]; j++ )
            *X(0, j+1) = *inputs(k, j);

    // Calcul de chaque neurone 
    for( int l = 1; l < L; l++ )
        for( int j = 1; j <= d[l]; j++ )
        {
            float signal = 0;
            int i = 0;

            for( ; i <= d[l-1]-8; i += 8 )
            {
                float signal0 = *W(l, i, j) * *X(l-1, i);
                float signal1 = *W(l, i+1, j) * *X(l-1, i+1);
                float signal2 = *W(l, i+2, j) * *X(l-1, i+2);
                float signal3 = *W(l, i+3, j) * *X(l-1, i+3);
                float signal4 = *W(l, i+4, j) * *X(l-1, i+4);
                float signal5 = *W(l, i+5, j) * *X(l-1, i+5);
                float signal6 = *W(l, i+6, j) * *X(l-1, i+6);
                float signal7 = *W(l, i+7, j) * *X(l-1, i+7);
                float signal01 = signal0 + signal1;
                float signal23 = signal2 + signal3;
                float signal45 = signal4 + signal5;
                float signal67 = signal6 + signal7;
                float signal0123 = signal01 + signal23;
                float signal4567 = signal45 + signal67;
                signal += signal0123 + signal4567;
            }

            for( ; i <= d[l-1]; i++ )
                signal += *W(l, i, j) * *X(l-1, i);
            
            if( is_used_for_classification || l != L-1 )
                signal = tanh(signal);
            
            *X(l, j) = signal;
        }
}

void MLP::retropropagate( int k, float alpha )
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
            int j = 1;

            for( ; j <= d[l]-8; j += 8 )
            {
                float total0 = *W(l, i, j) * *delta(l, j);
                float total1 = *W(l, i, j+1) * *delta(l, j+1);
                float total2 = *W(l, i, j+2) * *delta(l, j+2);
                float total3 = *W(l, i, j+3) * *delta(l, j+3);
                float total4 = *W(l, i, j+4) * *delta(l, j+4);
                float total5 = *W(l, i, j+5) * *delta(l, j+5);
                float total6 = *W(l, i, j+6) * *delta(l, j+6);
                float total7 = *W(l, i, j+7) * *delta(l, j+7);
                float total01 = total0 + total1;
                float total23 = total2 + total3;
                float total45 = total4 + total5;
                float total67 = total6 + total7;
                float total0123 = total01 + total23;
                float total4567 = total45 + total67;
                total += total0123 + total4567;
            }

            for( ; j <= d[l]; j++ )
                total += *W(l, i, j) * *delta(l, j);

           *delta(l-1,i) = ( 1.0 - pow(*X(l-1, i), 2.0) ) * total;
        }
        
    // Correction des poids  
    for( int l = 1; l < L; l++ )
        for( int i = 0; i <= d[l-1]; i++ )
        {
            int j = 1; 

            for( ;j <= d[l]-8; j += 8 )
            {
                *W(l, i, j) -= alpha * *X(l-1, i) * *delta(l, j);
                *W(l, i, j+1) -= alpha * *X(l-1, i) * *delta(l, j+1);
                *W(l, i, j+2) -= alpha * *X(l-1, i) * *delta(l, j+2);
                *W(l, i, j+3) -= alpha * *X(l-1, i) * *delta(l, j+3);
                *W(l, i, j+4) -= alpha * *X(l-1, i) * *delta(l, j+4);
                *W(l, i, j+5) -= alpha * *X(l-1, i) * *delta(l, j+5);
                *W(l, i, j+6) -= alpha * *X(l-1, i) * *delta(l, j+6);
                *W(l, i, j+7) -= alpha * *X(l-1, i) * *delta(l, j+7);
            }

            for( ;j <= d[l]; j++ )
                *W(l, i, j) -= alpha * *X(l-1, i) * *delta(l, j);
        }
}

void MLP::train( int nb_iterations, float alpha, int MSE_interval )
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

        propagate( k );
        retropropagate( k, alpha );

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

void MLP::quickTrain()
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
    std::cout << eigenInputs << std::endl;
    std::cout << eigenExpected_outputs << std::endl;
    std::cout << weights << std::endl;

    for( int i = 0; i <= d[0]; i++ )
        for( int j = 1; j <= d[L-1]; j++ )
            *W(1, i, j) = weights(i, j-1);
}

////////////////////////////////////////////////////////////////////////////////////////////// PREDICTING METHODS

void MLP::generatePrediction( int count, ... )
{
    if( count != d[0] )
        throw std::runtime_error(std::string("count n'est pas égal à " + std::to_string(d[0]) + " comme il devrait l'être ( " + std::to_string(count) + " donné à " + __FUNCTION__ + "() )"));

    va_list args;
    va_start( args, count );

    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = va_arg( args, double );

    va_end( args ); 

    propagate( -1 );
}

void MLP::generatePredictionArray( int count, float* array )
{
    if( count != d[0] )
        throw std::runtime_error(std::string("count n'est pas égal à " + std::to_string(d[0]) + " comme il devrait l'être ( " + std::to_string(count) + " donné à " + __FUNCTION__ + "() )"));

    for( int i = 1; i <= d[0]; i++ )
        *X(0, i) = array[i-1];

    propagate( -1 );
}

float MLP::getPrediction( int index )
{
    if( index < 0 || index >= d[L-1] )
        throw std::runtime_error(std::string("index donné n'est pas entre 0 et " + std::to_string(d[L-1]-1) + " comme il devrait l'être ( " + std::to_string(index) + " donné à " + __FUNCTION__ + "() )"));

    return *X(L-1, index+1);
}

float MLP::test()
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