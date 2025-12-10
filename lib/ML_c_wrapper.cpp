#include "LinearModel.hpp"
#include "MLP.hpp"
#include "RBF.hpp"

#ifdef _WIN32
    #define ML_EXPORT __declspec(dllexport)
#endif

extern "C" 
{
    ML_EXPORT void* LM_new( int nb_neurons_input_layer )
    {
        return reinterpret_cast<void*>(new LinearModel(nb_neurons_input_layer));
    }

    ML_EXPORT void* LM_copy( void* obj )
    {
        LinearModel* original = reinterpret_cast<LinearModel*>(obj);
        LinearModel* copy = new LinearModel(*original);
        return reinterpret_cast<void*>(copy);
    }

    ML_EXPORT void LM_delete( void* obj )
    {
        delete reinterpret_cast<LinearModel*>(obj);
    }

    ML_EXPORT void LM_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<LinearModel*>(obj)->setUsedForClassification( val ); 
    }

    ML_EXPORT void LM_initElements( void* obj, int count )
    {
        reinterpret_cast<LinearModel*>(obj)->initElements( count ); 
    }

    ML_EXPORT void LM_initElementsTest( void* obj, int count )
    {
        reinterpret_cast<LinearModel*>(obj)->initElementsTest( count );
    }

    ML_EXPORT void LM_addElement(void* obj, ... )
    { 
        int count = reinterpret_cast<LinearModel*>(obj)->getNbInputNeurons()+1;
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 
        reinterpret_cast<LinearModel*>(obj)->addElementArray( array ); 
        delete[] array;
    }

    ML_EXPORT void LM_addElementArray(void* obj, void* array )
    { 
        reinterpret_cast<LinearModel*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT void LM_addElementTestArray(void* obj, void* array )
    {
        reinterpret_cast<LinearModel*>(obj)->addElementTestArray( static_cast<float*>(array) );
    }

    ML_EXPORT void LM_print( void* obj, bool printX, bool printY, bool printW, bool printMSE )
    { 
        reinterpret_cast<LinearModel*>(obj)->print(printX, printY, printW, printMSE);
    }

    ML_EXPORT void LM_train( void* obj, int nb_iterations, float alpha, int MSE_interval )
    { 
        reinterpret_cast<LinearModel*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    ML_EXPORT void LM_quickTrain( void* obj )
    {
        reinterpret_cast<LinearModel*>(obj)->quickTrain(); 
    }

    ML_EXPORT float LM_predict(void* obj, ... )
    { 
        int count = reinterpret_cast<LinearModel*>(obj)->getNbInputNeurons();
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 

        float res = reinterpret_cast<LinearModel*>(obj)->predictArray( array ); 

        delete[] array;

        return res;
    }

    ML_EXPORT float LM_predictArray(void* obj, void* array )
    { 
        return reinterpret_cast<LinearModel*>(obj)->predictArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT float LM_test( void* obj )
    {
        return reinterpret_cast<LinearModel*>(obj)->test();
    }

    ML_EXPORT float LM_realTest( void* obj )
    {
        return reinterpret_cast<LinearModel*>(obj)->realTest();
    }

    ML_EXPORT int LM_getMSESize( void* obj )
    { 
        return reinterpret_cast<LinearModel*>(obj)->getMSESize(); 
    }

    ML_EXPORT float LM_MSE( void* obj, int index )
    { 
        return reinterpret_cast<LinearModel*>(obj)->MSE( index ); 
    }

    ML_EXPORT int LM_getNbInputNeurons( void* obj )
    {
        return reinterpret_cast<LinearModel*>(obj)->getNbInputNeurons();
    }

    ML_EXPORT float LM_getWeight( void* obj, int index)
    {
        return reinterpret_cast<LinearModel*>(obj)->getWeight( index );
    }

    ML_EXPORT void LM_setWeights( void* obj, void* weights)
    {
        reinterpret_cast<LinearModel*>(obj)->setWeights( static_cast<float*>(weights) );
    }
}

extern "C" 
{
    // Le constructeur dans le wrapper ne peut être variadique
    ML_EXPORT void* MLP_new( int count, ... )
    {
        int* array = new int[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = va_arg( args, int );
        }
        va_end( args ); 

        return reinterpret_cast<void*>(new MLP(count, array));
        delete[] array;
    }
    
    ML_EXPORT void* MLP_new_array( int count, void* d )
    {
        return reinterpret_cast<void*>(new MLP(count, static_cast<int*>(d)));
    }

    ML_EXPORT void MLP_delete( void* obj )
    {
        delete reinterpret_cast<MLP*>(obj);
    }

    ML_EXPORT void MLP_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<MLP*>(obj)->setUsedForClassification( val ); 
    }

    ML_EXPORT void MLP_initElements( void* obj, int count )
    {
        reinterpret_cast<MLP*>(obj)->initElements( count ); 
    }

    ML_EXPORT void MLP_addElement(void* obj, ... )
    { 
        int count = reinterpret_cast<MLP*>(obj)->getNbInputNeurons()+reinterpret_cast<MLP*>(obj)->getNbOutputNeurons();
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 
        reinterpret_cast<MLP*>(obj)->addElementArray( array ); 
        delete[] array;
    }

    ML_EXPORT void MLP_addElementArray(void* obj, void* array )
    { 
        reinterpret_cast<MLP*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT void MLP_print( void* obj, int nbElementsToPrint )
    { 
        reinterpret_cast<MLP*>(obj)->print(nbElementsToPrint);
    }

    ML_EXPORT void MLP_train( void* obj, int nb_iterations, float alpha, int MSE_interval )
    { 
        reinterpret_cast<MLP*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    ML_EXPORT void MLP_quickTrain( void* obj )
    {
        reinterpret_cast<MLP*>(obj)->quickTrain(); 
    }

    ML_EXPORT void MLP_generatePrediction(void* obj, ... )
    { 
        int count = reinterpret_cast<MLP*>(obj)->getNbInputNeurons();
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 

        reinterpret_cast<MLP*>(obj)->generatePredictionArray( array ); 

        delete[] array;
    }

    ML_EXPORT void MLP_generatePredictionArray(void* obj, void* array )
    { 
        reinterpret_cast<MLP*>(obj)->generatePredictionArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT float MLP_getPrediction( void* obj, int index )
    { 
        return reinterpret_cast<MLP*>(obj)->getPrediction( index ); 
    }

    ML_EXPORT float MLP_test( void* obj )
    {
        return reinterpret_cast<MLP*>(obj)->test();
    }

    ML_EXPORT int MLP_getMSESize( void* obj )
    { 
        return reinterpret_cast<MLP*>(obj)->getMSESize(); 
    }

    ML_EXPORT float MLP_MSE( void* obj, int index )
    { 
        return reinterpret_cast<MLP*>(obj)->MSE( index ); 
    }

    ML_EXPORT int MLP_getNbInputNeurons( void* obj)
    {
        return reinterpret_cast<MLP*>(obj)->getNbInputNeurons();
    }

    ML_EXPORT int MLP_getNbOutputNeurons( void* obj)
    {
        return reinterpret_cast<MLP*>(obj)->getNbOutputNeurons();
    }

    ML_EXPORT int MLP_getL( void* obj)
    {
        return reinterpret_cast<MLP*>(obj)->getL();
    }

    ML_EXPORT int MLP_getD( void* obj, int index)
    {
        return reinterpret_cast<MLP*>(obj)->getD(index);
    }

    ML_EXPORT float MLP_getW( void* obj, int layer, int neuron_out, int neuron_in)
    {
        return reinterpret_cast<MLP*>(obj)->getW(layer, neuron_out, neuron_in);
    }

    ML_EXPORT void MLP_setW( void* obj, int layer, int neuron_out, int neuron_in, float weight)
    {
        return reinterpret_cast<MLP*>(obj)->setW(layer, neuron_out, neuron_in, weight);
    }
}

extern "C" 
{
    ML_EXPORT void* RBF_new( int nb_neurons_input_layer, float gamma )
    {
        return reinterpret_cast<void*>(new RBF(nb_neurons_input_layer, gamma));
    }

    ML_EXPORT void RBF_delete( void* obj )
    {
        delete reinterpret_cast<RBF*>(obj);
    }

    ML_EXPORT void RBF_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<RBF*>(obj)->setUsedForClassification( val ); 
    }

    ML_EXPORT void RBF_initElements( void* obj, int count )
    {
        reinterpret_cast<RBF*>(obj)->initElements( count ); 
    }

    ML_EXPORT void RBF_addElement(void* obj, ... )
    { 
        int count = reinterpret_cast<RBF*>(obj)->getNbInputNeurons()+1;
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 
        reinterpret_cast<RBF*>(obj)->addElementArray( array ); 
        delete[] array;
    }

    ML_EXPORT void RBF_addElementArray(void* obj, void* array )
    { 
        reinterpret_cast<RBF*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT void RBF_print( void* obj )
    { 
        reinterpret_cast<RBF*>(obj)->print(); 
    }

    ML_EXPORT void RBF_generateClusters( void* obj, int nb_clusters, int nb_iterations )
    { 
        reinterpret_cast<RBF*>(obj)->generateClusters( nb_clusters, nb_iterations ); 
    }

    ML_EXPORT void RBF_train( void* obj )
    { 
        reinterpret_cast<RBF*>(obj)->train(); 
    }

    ML_EXPORT float RBF_predict(void* obj, ... )
    { 
        int count = reinterpret_cast<RBF*>(obj)->getNbInputNeurons();
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 

        float res = reinterpret_cast<RBF*>(obj)->predictArray( array ); 

        delete[] array;

        return res;
    }

    ML_EXPORT float RBF_predictArray(void* obj, void* array )
    { 
        return reinterpret_cast<RBF*>(obj)->predictArray( static_cast<float*>(array) ); 
    }

    ML_EXPORT float RBF_test( void* obj )
    {
        return reinterpret_cast<RBF*>(obj)->test();
    }

    ML_EXPORT int RBF_getNbCluster( void* obj )
    { 
        return reinterpret_cast<RBF*>(obj)->getNbCluster(); 
    }

    ML_EXPORT float RBF_getClusterElement( void* obj, int index, int element )
    { 
        return reinterpret_cast<RBF*>(obj)->getClusterElement( index, element ); 
    }
}