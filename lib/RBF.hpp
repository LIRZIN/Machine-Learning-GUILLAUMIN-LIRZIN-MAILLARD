#ifndef RBF_HPP
#define RBF_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

class RBF
{
    // n-th column : n-th element
    // n-th line : n-th component of an element
    // Eigen::MatrixXf<type, nb_rows, nb_columns>
    // typedef Matrix<float, Dynamic, 1> VectorXf;
    int nb_components_of_element;
    int nb_elements = 0;
    int nb_clusters = 0;
    bool is_used_for_classification = true;

    Eigen::MatrixXf input;
    Eigen::MatrixXf clusters;
    Eigen::VectorXf output;
    Eigen::VectorXf weights;

    float gamma;
    
    float dist( Eigen::VectorXf& a, Eigen::VectorXf& b );
    Eigen::VectorXf getGaussVector( Eigen::VectorXf& X );

    public : 
        RBF( int init_nb_components_of_element, float init_gamma );

        void setUsedForClassification( bool val ) { is_used_for_classification = val; }

        void initElements( int nbElements );
        void addElement( int count, ... );
        void addElementArray( float* array );

        void print();

        float predict( int count, ... );
        float predictArray( float* array );
        float predictVector( Eigen::VectorXf& X_k );

        void generateClusters( int nb_clusters, int nb_iterations );
        int getNbCluster() { return nb_clusters; }
        float getClusterElement( int index, int element ) { return clusters(index, element); }

        //void train( int nb_iterations, float alpha, int MSE_interval = 0 );
        void train();

        // Génère une prédiction pour chaque couple ( entrée, sortie attendue ) 
        // et calcule un pourcentage de succès selon les valeurs générées
        float test();

        int getNbInputNeurons() { return nb_components_of_element; }
};

#endif