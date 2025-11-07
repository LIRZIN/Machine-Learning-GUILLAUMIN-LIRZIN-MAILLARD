#include "MLP.hpp"

extern "C" 
{
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

    void MLP_initElements( void* obj, int count )
    {
        reinterpret_cast<MLP*>(obj)->initElements( count ); 
    }

    void MLP_addElement(void* obj, int count, ... ) 
    { 
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 
        reinterpret_cast<MLP*>(obj)->addElementArray( count, array ); 
        delete[] array;
    }

    void MLP_addElementArray(void* obj, int count, void* array ) 
    { 
        reinterpret_cast<MLP*>(obj)->addElementArray( count, static_cast<float*>(array) ); 
    }

    void MLP_print( void* obj ) 
    { 
        reinterpret_cast<MLP*>(obj)->print(); 
    }

    void MLP_train( void* obj, int nb_iterations, float alpha, bool is_used_for_classification, int MSE_interval )
    { 
        reinterpret_cast<MLP*>(obj)->train( nb_iterations, alpha, is_used_for_classification, MSE_interval ); 
    }

    void MLP_generatePrediction(void* obj, int count, ... ) 
    { 
        float* array = new float[count];

        va_list args;
        va_start( args, count );
        for( int i = 0; i < count; i++ )
        {
            array[i] = static_cast<float>(va_arg( args, double ));
        }
        va_end( args ); 

        reinterpret_cast<MLP*>(obj)->generatePredictionArray( true, count, array ); 

        delete[] array;
    }

    void MLP_generatePredictionArray(void* obj, int count, void* array ) 
    { 
        reinterpret_cast<MLP*>(obj)->generatePredictionArray( true, count, static_cast<float*>(array) ); 
    }

    float MLP_getPrediction( void* obj, int index )
    { 
        return reinterpret_cast<MLP*>(obj)->getPrediction( index ); 
    }

    float MLP_test( void* obj, bool is_used_for_classification )
    {
        return reinterpret_cast<MLP*>(obj)->test(is_used_for_classification);
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

/*
lib = ctypes.CDLL("/content/MLP.so")

lib.MLP_new.restype = ctypes.c_void_p
lib.MLP_new.argtypes = [ctypes.c_int]
lib.MLP_delete.argtypes = [ctypes.c_void_p]

lib.MLP_initElements.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.MLP_addElement.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.MLP_printElements.argtypes = [ctypes.c_void_p]

lib.MLP_train.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_bool, ctypes.c_int]
lib.MLP_generatePrediction.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int]
lib.MLP_getPrediction.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.MLP_getPrediction.restype = ctypes.c_float
lib.MLP_test.restype = ctypes.c_float
lib.MLP_test.argtypes = [ctypes.c_bool]

lib.MLP_getMSESize.argtypes = [ctypes.c_void_p]
lib.MLP_getMSESize.restype = ctypes.c_int
lib.MLP_MSE.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.MLP_MSE.restype = ctypes.c_float

mlp = lib.MLP_new( 2, 2, 1 )

for i in range(3):
  lib.MLP_addElement( mlp, 3, ctypes.c_double(X[i][0]), ctypes.c_double(X[i][1]), ctypes.c_double(Y[i]) )
'''
lib.MLP_train( mlp, 100000, 0.001, True, 0 )

all_grid_points = []
all_grid_points_colors = []
for x1 in range(100, 300):
  for x2 in range(100, 300):
    x1_p = x1 / 100.0
    x2_p = x2 / 100.0
    lib.MLP_generatePrediction( mlp, 2, ctypes.c_double(x1_p), ctypes.c_double(x2_p) )
    predicted_value = lib.MLP_getPrediction(mlp, 0)
    all_grid_points.append([x1_p, x2_p])
    all_grid_points_colors.append('lightblue' if predicted_value >= 0 else 'pink')

all_grid_points = np.array(all_grid_points)
lib.MLP_delete(mlp)
'''

*/