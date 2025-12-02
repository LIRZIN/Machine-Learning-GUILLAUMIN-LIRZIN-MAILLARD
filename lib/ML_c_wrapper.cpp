#include "LinearModel.hpp"
#include "MLP.hpp"
#include "RBF.hpp"

extern "C" 
{
    void* LM_new( int nb_neurons_input_layer )
    {
        return reinterpret_cast<void*>(new LinearModel(nb_neurons_input_layer));
    }

    void LM_delete( void* obj )
    {
        delete reinterpret_cast<LinearModel*>(obj);
    }

    void LM_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<LinearModel*>(obj)->setUsedForClassification( val ); 
    }

    void LM_initElements( void* obj, int count )
    {
        reinterpret_cast<LinearModel*>(obj)->initElements( count ); 
    }

    void LM_addElement(void* obj, ... ) 
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

    void LM_addElementArray(void* obj, void* array ) 
    { 
        reinterpret_cast<LinearModel*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    void LM_print( void* obj ) 
    { 
        reinterpret_cast<LinearModel*>(obj)->print(); 
    }

    void LM_train( void* obj, int nb_iterations, float alpha, int MSE_interval )
    { 
        reinterpret_cast<LinearModel*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    void LM_quickTrain( void* obj )
    {
        reinterpret_cast<LinearModel*>(obj)->quickTrain(); 
    }

    float LM_predict(void* obj, ... ) 
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

    float LM_predictArray(void* obj, void* array ) 
    { 
        return reinterpret_cast<LinearModel*>(obj)->predictArray( static_cast<float*>(array) ); 
    }

    float LM_test( void* obj )
    {
        return reinterpret_cast<LinearModel*>(obj)->test();
    }

    int LM_getMSESize( void* obj )
    { 
        return reinterpret_cast<LinearModel*>(obj)->getMSESize(); 
    }

    float LM_MSE( void* obj, int index )
    { 
        return reinterpret_cast<LinearModel*>(obj)->MSE( index ); 
    }
}

extern "C" 
{
    // Le constructeur dans le wrapper ne peut être variadique
    void* MLP_new( int count, ... )
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
    
    void* MLP_new_array( int count, void* d )
    {
        return reinterpret_cast<void*>(new MLP(count, static_cast<int*>(d)));
    }

    void MLP_delete( void* obj )
    {
        delete reinterpret_cast<MLP*>(obj);
    }

    void MLP_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<MLP*>(obj)->setUsedForClassification( val ); 
    }

    void MLP_initElements( void* obj, int count )
    {
        reinterpret_cast<MLP*>(obj)->initElements( count ); 
    }

    void MLP_addElement(void* obj, ... ) 
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

    void MLP_addElementArray(void* obj, void* array ) 
    { 
        reinterpret_cast<MLP*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    void MLP_print( void* obj ) 
    { 
        reinterpret_cast<MLP*>(obj)->print(); 
    }

    void MLP_train( void* obj, int nb_iterations, float alpha, int MSE_interval )
    { 
        reinterpret_cast<MLP*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    void MLP_quickTrain( void* obj )
    {
        reinterpret_cast<MLP*>(obj)->quickTrain(); 
    }

    void MLP_generatePrediction(void* obj, ... ) 
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

    void MLP_generatePredictionArray(void* obj, void* array ) 
    { 
        reinterpret_cast<MLP*>(obj)->generatePredictionArray( static_cast<float*>(array) ); 
    }

    float MLP_getPrediction( void* obj, int index )
    { 
        return reinterpret_cast<MLP*>(obj)->getPrediction( index ); 
    }

    float MLP_test( void* obj )
    {
        return reinterpret_cast<MLP*>(obj)->test();
    }

    int MLP_getMSESize( void* obj )
    { 
        return reinterpret_cast<MLP*>(obj)->getMSESize(); 
    }

    float MLP_MSE( void* obj, int index )
    { 
        return reinterpret_cast<MLP*>(obj)->MSE( index ); 
    }
}

extern "C" 
{
    void* RBF_new( int nb_neurons_input_layer, float gamma )
    {
        return reinterpret_cast<void*>(new RBF(nb_neurons_input_layer, gamma));
    }

    void RBF_delete( void* obj )
    {
        delete reinterpret_cast<RBF*>(obj);
    }

    void RBF_setUsedForClassification( void* obj, bool val )
    {
        reinterpret_cast<RBF*>(obj)->setUsedForClassification( val ); 
    }

    void RBF_initElements( void* obj, int count )
    {
        reinterpret_cast<RBF*>(obj)->initElements( count ); 
    }

    void RBF_addElement(void* obj, ... ) 
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

    void RBF_addElementArray(void* obj, void* array ) 
    { 
        reinterpret_cast<RBF*>(obj)->addElementArray( static_cast<float*>(array) ); 
    }

    void RBF_print( void* obj ) 
    { 
        reinterpret_cast<RBF*>(obj)->print(); 
    }

    void RBF_generateClusters( void* obj, int nb_clusters, int nb_iterations )
    { 
        reinterpret_cast<RBF*>(obj)->generateClusters( nb_clusters, nb_iterations ); 
    }

    void RBF_train( void* obj )
    { 
        reinterpret_cast<RBF*>(obj)->train(); 
    }

    float RBF_predict(void* obj, ... ) 
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

    float RBF_predictArray(void* obj, void* array ) 
    { 
        return reinterpret_cast<RBF*>(obj)->predictArray( static_cast<float*>(array) ); 
    }

    float RBF_test( void* obj )
    {
        return reinterpret_cast<RBF*>(obj)->test();
    }

    int RBF_getNbCluster( void* obj )
    { 
        return reinterpret_cast<RBF*>(obj)->getNbCluster(); 
    }

    float RBF_getClusterElement( void* obj, int index, int element )
    { 
        return reinterpret_cast<RBF*>(obj)->getClusterElement( index, element ); 
    }
}