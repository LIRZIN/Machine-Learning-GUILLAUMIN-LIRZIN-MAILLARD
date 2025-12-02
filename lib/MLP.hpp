#ifndef MLP_HPP
#define MLP_HPP

#define _CRT_NO_VA_START_VALIDATION

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <string>
#include <Eigen/Dense>

class MLP
{
    // vecteur utilisé seulement dans le constructeur variadique. Ne sert à rien autrement
    std::vector<int> vector_d;
    // dicte si le réseau de neurones est utilisé pour classifier ou non
    bool is_used_for_classification = true;
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

    // Initialise la mémoire des matrices _W, _X, _delta 
    // _W est initialisée avec des valeurs aléatoires et _X et _delta sont initialisées avec des 0 
    // ( à part le premier neurone de chaque couche qui est mis à 1 )
    void init_matrices();

    // fonctions de lecture des matrices
    // ne jamais utiliser les tableaux _* directement, il faut appeler ces fonctions ( à part getNeuronsData )
    // retourne un pointeur vers l'élément recherché pour lire et modifier l'élément à son emplacement

    // Retourne le poids contenu dans la flèche allant de neuron_out de layer-1 vers le neuron_in de layer 
    // le biais 1 ne prends pas de flèche en entrée donc neuron_in ne peut être égal à 0
    float* W( int layer, int neuron_out, int neuron_in );
    // Retourne soit la valeur ( X ) ou le delta -( delta ) du neurone neuron dans la couche layer
    // l'indice 0 de chaque couche dans X renvoit le biais 1
    float* getNeuronsData( float* array, int layer, int neuron );
    float* X( int layer, int neuron );
    float* delta( int layer, int neuron );

    // Retourne les entrées et les sorties attendues
    // Les entrées ne contiennent pas le biais 1 de la couche d'entrée
    float* inputs( int elemIndex, int componentIndex = 0 );
    float* expected_outputs( int elemIndex, int componentIndex = 0 );

    // Remplie X de couche en couche pour calculer la sortie du réseau
    void propagate( int k );
    // Remplie delta de couche en couche pour ensuite rectifier les poids
    void retropropagate( int k, float alpha );

    public : 
        template<typename... Ints>
        MLP(Ints... values) : vector_d{ int(values)... } 
        {
            L = vector_d.size();
            if( L < 2 )
                throw std::runtime_error(std::string("Le nombre de couche donné doit être supérieur ou égal à 2 ( " + std::to_string(L) + " couches données au constructeur variadique )"));

            d = new int[L];
            for( int l = 0; l < L; l++ )
                d[l] = vector_d[l];
            vector_d.clear();
            init_matrices();
        }

        MLP( int count, int* d_init )  
        {
            if( count < 2 )
                throw std::runtime_error(std::string("Le nombre de couche donné doit être supérieur ou égal à 2 ( " + std::to_string(count) + " couches données au constructeur traditionel )"));

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

        void setUsedForClassification( bool val ) { is_used_for_classification = val; }
        // Imprime le contenu des toutes les matrices 
        void print();

        // Avant de donner des éléments au réseau, il faut déclarer le nombre d'éléments que l'utilisateur compte lui donner
        // Cela évite de faire de ré-allocation mémoire à chaque fois qu'on ajoute un élément
        void initElements( int nb_elements_alloc );
        // Ajoute un élément aux inputs et expected_outputs
        void addElement( int count, ... );
        void addElementArray( float* array );

        // Entraine le réseau sur nb_iterations
        void train( int nb_iterations, float alpha, int MSE_interval = 0 );
        void quickTrain();

        // Génère une prédiction selon l'entrée donnée et calcule la sortie qui est stocké dans la dernière couche de X
        void generatePrediction( int count, ... );
        void generatePredictionArray( float* array );
        // Renvoie la valeur du neurone à l'index+1 de la dernière couche de X
        float getPrediction( int index );

        // Génère une prédiction pour chaque couple ( entrée, sortie attendue ) 
        // et calcule un pourcentage de succès selon les valeurs générées
        float test();

        // Simple getters pour lire les valeurs de MSE
        int getMSESize() { return nb_MSE; }
        float MSE( int index ) { return _MSE[index]; }

        int getNbInputNeurons() { return d[0]; }
        int getNbOutputNeurons() { return d[L-1]; }
};

#endif