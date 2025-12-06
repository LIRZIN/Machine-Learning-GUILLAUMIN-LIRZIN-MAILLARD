#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

class LinearModel
{
    // n-th column : n-th element
    // n-th line : n-th component of an element
    // Eigen::MatrixXf<type, nb_rows, nb_columns>
    // typedef Matrix<float, Dynamic, 1> VectorXf;
    int nb_components_of_element;
    int nb_elements = 0;
    int nb_elementsTest = 0;
    bool is_used_for_classification = true;

    Eigen::MatrixXf input;
    Eigen::VectorXf output;

    Eigen::MatrixXf inputTest;
    Eigen::VectorXf outputTest;

    Eigen::VectorXf weights;

    // Stocke les MSE généré pendant un entrainement 
    Eigen::VectorXf _MSE;

    public :
        LinearModel( int init_nb_components_of_element );

        void setUsedForClassification( bool val ) { is_used_for_classification = val; }

        void initElements( int nbElements );
        void initElementsTest( int nbElements );
        void addElement(int count, ... );
        void addElementArray( float* array );
        void addElementTestArray( float* array );

        void print(bool printX, bool printY, bool printW, bool printMSE);

        float predict( int count, ... );
        float predictArray( float* array );
        float predictVector( Eigen::VectorXf& X_k_with_one );

        void train( int nb_iterations, float alpha, int MSE_interval = 0 );
        void quickTrain();

        // Génère une prédiction pour chaque couple ( entrée, sortie attendue ) 
        // et calcule un pourcentage de succès selon les valeurs générées
        float test();
        float realTest();

        // Simple getters pour lire les valeurs de MSE
        int getMSESize() { return _MSE.size(); }
        float MSE( int index ) { return _MSE[index]; }

        int getNbInputNeurons() { return nb_components_of_element; }
};

#endif