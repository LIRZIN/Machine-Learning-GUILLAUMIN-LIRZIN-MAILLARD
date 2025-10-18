#ifndef MULTI_LEVEL_PERCEPTRON_HPP
#define MULTI_LEVEL_PERCEPTRON_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

class MultiLevelPerceptron
{
    int nb_neurons_in_input_layer; // doesn't include the constant 1 neuron added internally
    int nb_hidden_layers;
    int nb_neurons_in_hidden_layer; // doesn't include the constant 1 neuron added internally
    int nb_neurons_in_output_layer;

    int nb_elements = 0;
    int nb_input_set_weights = 0;
    int nb_hidden_set_weights = 0;

    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd input_weights;
    Eigen::MatrixXd hidden_weights;

    /*
    int MSE_interval;
    std::vector<float> MSEs;
    */

    void init_matrices();

    // In Weights : weights that are used to compute a neuron's value 
    // Out Weights : weights that are applied to a neuron to compute the next layer's neurons
    Eigen::VectorXd getInWeights( int layer, int neuron );
    void setInWeights( int layer, int neuron, Eigen::VectorXd& weights );
    Eigen::VectorXd getOutWeights( int layer, int neuron );
    void setOutWeights( int layer, int neuron, Eigen::VectorXd& weights );

    void compute_neuron_values( Eigen::VectorXd& input_k, Eigen::MatrixXd& computed_hidden_neurons, Eigen::VectorXd& computed_output_neurons );
    void update_weights( Eigen::VectorXd& input_k, Eigen::VectorXd& output_k, Eigen::MatrixXd& computed_hidden_neurons, Eigen::VectorXd& computed_output_neurons, double alpha  );

    public : 
        MultiLevelPerceptron( int nb_neurons_in_input_layer, int nb_neurons_in_output_layer ) 
        : nb_neurons_in_input_layer(nb_neurons_in_input_layer), nb_hidden_layers(0), nb_neurons_in_hidden_layer(0), nb_neurons_in_output_layer(nb_neurons_in_output_layer)
        {
            init_matrices();
        }

        MultiLevelPerceptron( int nb_neurons_in_input_layer, int nb_layers, int nb_neurons_in_output_layer ) 
        : nb_neurons_in_input_layer(nb_neurons_in_input_layer), nb_hidden_layers(nb_layers-1), nb_neurons_in_hidden_layer(nb_neurons_in_input_layer), nb_neurons_in_output_layer(nb_neurons_in_output_layer)
        {
            init_matrices();
        }

        MultiLevelPerceptron( int nb_neurons_in_input_layer, int nb_layers, int nb_neurons_in_hidden_layer, int nb_neurons_in_output_layer ) 
        : nb_neurons_in_input_layer(nb_neurons_in_input_layer), nb_hidden_layers(nb_layers-1), nb_neurons_in_hidden_layer(nb_neurons_in_hidden_layer), nb_neurons_in_output_layer(nb_neurons_in_output_layer)
        {
            init_matrices();
        }

        void addElement( int count, ... );
        void printElements();

        void train( int nb_iterations, double alpha );
        Eigen::VectorXd predict( double x1, double x2 ); // test method
};

#endif