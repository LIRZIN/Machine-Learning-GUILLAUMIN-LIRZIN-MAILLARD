#ifndef MLP_HPP
#define MLP_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

class MLP
{
    // nombre de neurones par couche, inclus la couche d'entrée
    int* d = NULL;
    // nombre de couches
    int L = 0;
    // poids dans un tableau 1D
    float* _W = NULL;
    // neurones dans un tableau 1D
    float* _X = NULL;
    // delta dans un tableau 1D
    float* _delta = NULL;

    // nombre d'éléments d'entrainement
    int nb_elements = 0;
    // entrées d'entrainement dans un tableau 1D
    float* _inputs = NULL;
    // sorties d'entrainement dans un tableau 1D
    float* _expected_outputs = NULL;

    // Stocke les MSE généré pendant un entrainement 
    int nb_MSE = 0;
    float* _MSE = NULL;

    void init_matrices();

    float* W( int layer, int neuron_out, int neuron_in );
    float* getNeuronsData( float* array, int layer, int neuron );
    float* X( int layer, int neuron );
    float* delta( int layer, int neuron );
    float* inputs( int elemIndex, int componentIndex = 0 );
    float* expected_outputs( int elemIndex, int componentIndex = 0 );

    void propagate( int k, bool is_used_for_classification );
    void retropropagate( int k, bool is_used_for_classification, float alpha );

    public : 
        MLP( int count, ... )  
        {
            if( count < 2 )
            {
                // exception
            }

            L = count;
            d = new int[L];
            va_list args;
            va_start( args, count );

            for( int l = 0; l < L; l++ )
                d[l] = va_arg( args, double );

            va_end( args ); 
            init_matrices();
        }

        MLP( int count, int* d_init )  
        {
            if( count < 2 )
            {
                // exception
            }

            L = count;
            d = new int[L];
            for( int l = 0; l < L; l++ )
                d[l] = d_init[l];

            init_matrices();
        }

        ~MLP()
        {
            delete[] d;
            delete[] _W;
            delete[] _X;
            delete[] _delta;
            delete[] _inputs;
            delete[] _expected_outputs;
            delete[] _MSE;
        }

        void initElements( int nb_elements_alloc );
        void addElement( int count, ... );
        void addElementArray( int count, float* array );
        void printElements();

        void train( int nb_iterations, float alpha, bool is_used_for_classification, int MSE_interval = 0 );

        void generatePrediction( bool is_used_for_classification, int count, ... );
        void generatePredictionArray( bool is_used_for_classification, int count, float* array );
        float getPrediction( int index );

        float test( bool is_used_for_classification );

        int getMSESize() { return nb_MSE; }
        float MSE( int index ) { return _MSE[index]; }
};

#endif