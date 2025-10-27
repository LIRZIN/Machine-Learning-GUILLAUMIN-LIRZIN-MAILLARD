#include "MultiLevelPerceptron.hpp"

extern "C" 
{
    void* MultiLevelPerceptron_new_2(bool n, int a, int b) { return reinterpret_cast<void*>(new MultiLevelPerceptron(n, a, b)); }
    void* MultiLevelPerceptron_new_3(bool n, int a, int b, int c) { return reinterpret_cast<void*>(new MultiLevelPerceptron(n, a, b, c)); }
    void* MultiLevelPerceptron_new_4(bool n, int a, int b, int c, int d) { return reinterpret_cast<void*>(new MultiLevelPerceptron(n, a, b, c, d)); }

    void MultiLevelPerceptron_addElement(void* obj, int count, ... ) 
    { 
        double* array = new double[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = va_arg( args, double );
        }
        va_end( args ); 
        reinterpret_cast<MultiLevelPerceptron*>(obj)->addElementArray( count, array ); 
        delete array;
    }

    void MultiLevelPerceptron_printElements( void* obj ) 
    { 
        reinterpret_cast<MultiLevelPerceptron*>(obj)->printElements(); 
    }

    void MultiLevelPerceptron_train( void* obj, int nb_iterations, double alpha, int MSE_interval = 0 )
    { 
        reinterpret_cast<MultiLevelPerceptron*>(obj)->train( nb_iterations, alpha, MSE_interval ); 
    }

    void MultiLevelPerceptron_generatePrediction(void* obj, int count, ... ) 
    { 
        double* array = new double[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = va_arg( args, double );
        }
        va_end( args ); 

        reinterpret_cast<MultiLevelPerceptron*>(obj)->generatePredictionArray( count, array ); 

        delete array;
    }

    double MultiLevelPerceptron_getPrediction( void* obj, int index )
    { 
        return reinterpret_cast<MultiLevelPerceptron*>(obj)->getPrediction( index ); 
    }

    int MultiLevelPerceptron_getMSESize( void* obj )
    { 
        return reinterpret_cast<MultiLevelPerceptron*>(obj)->getMSESize(); 
    }

    double MultiLevelPerceptron_MSE( void* obj, int index )
    { 
        return reinterpret_cast<MultiLevelPerceptron*>(obj)->MSE( index ); 
    }
}