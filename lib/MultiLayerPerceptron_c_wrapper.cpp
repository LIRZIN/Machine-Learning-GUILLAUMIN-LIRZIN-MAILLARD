#include "MultiLayerPerceptron.hpp"

extern "C" 
{
    void* MultiLayerPerceptron_new_2(bool n, int a, int b) {  }
    void* MultiLayerPerceptron_new_3(bool n, int a, int b, int c) { return reinterpret_cast<void*>(new MultiLayerPerceptron(n, a, b, c)); }
    void* MultiLayerPerceptron_new_4(bool n, int a, int b, int c, int d) { return reinterpret_cast<void*>(new MultiLayerPerceptron(n, a, b, c, d)); }

    void MultiLayerPerceptron_addElement(void* obj, int count, ... ) 
    { 
        double* array = new double[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = va_arg( args, double );
        }
        va_end( args ); 
        reinterpret_cast<MultiLayerPerceptron*>(obj)->addElementArray( count, array ); 
        delete array;
    }

    void MultiLayerPerceptron_printElements( void* obj ) 
    { 
        reinterpret_cast<MultiLayerPerceptron*>(obj)->printElements(); 
    }

    void MultiLayerPerceptron_train( void* obj, int nb_iterations, double alpha, int MSE_interval = 0 )
    { 
        reinterpret_cast<MultiLayerPerceptron*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    void MultiLayerPerceptron_generatePrediction(void* obj, int count, ... ) 
    { 
        double* array = new double[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = va_arg( args, double );
        }
        va_end( args ); 

        reinterpret_cast<MultiLayerPerceptron*>(obj)->generatePredictionArray( count, array ); 

        delete array;
    }

    double MultiLayerPerceptron_getPrediction( void* obj, int index )
    { 
        return reinterpret_cast<MultiLayerPerceptron*>(obj)->getPrediction( index ); 
    }

    int MultiLayerPerceptron_getMSESize( void* obj )
    { 
        return reinterpret_cast<MultiLayerPerceptron*>(obj)->getMSESize(); 
    }

    double MultiLayerPerceptron_MSE( void* obj, int index )
    { 
        return reinterpret_cast<MultiLayerPerceptron*>(obj)->MSE( index ); 
    }
}