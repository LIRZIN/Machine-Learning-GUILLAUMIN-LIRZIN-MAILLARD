#ifndef MULTI_LEVEL_PERCEPTRON_HPP
#define MULTI_LEVEL_PERCEPTRON_HPP

#include <iostream>
#include <cstdarg>
#include <math.h>
#include <Eigen/Dense>

#define nb_neurons_in_input_layer (nb_neurons_in_layer[0])
#define nb_neurons_in_output_layer (nb_neurons_in_layer[nb_layers-1])

class MultiLayerPerceptron
{
    int* nb_neurons_in_layer;
    int nb_layers;

    // Contains the input data to train the network
    // It has [nb_neurons_in_input_layer+1] rows and [number_of_elements] columns
    Eigen::MatrixXd input;
    // Contains the output data to train the network
    // The output at the column i corresponds to the input at the columb i in the member [input]
    // It has [nb_neurons_in_output_layer] rows and [number_of_elements] columns
    Eigen::MatrixXd output;
    // Contains the weights applied to the input data when computing the neurons' value in the second layer of the neural network
    // It has [nb_neurons_in_input_layer+1] rows and [number_of_neurons_in_the_next_layer] columns
    // Contains the weights applied to a hidden layer's neurons when computing the neurons' value in the next layer of the neural network
    // It has [nb_neurons_in_hidden_layer+1] rows and [nb_neurons_in_hidden_layer*nb_hidden_layers+nb_neurons_in_output_layer] columns
    // If the network only contains 2 layers ( input and output layers ), this matrix is left unused
    Eigen::MatrixXd* weights;

    Eigen::VectorXd MSE_values;

    Eigen::VectorXd predictedOutput;

    /* Initializes the memory of each the class's matrices
     * If a matrix is to contain weights, each of its elements is initialized randomly between -1 and 1
     */
    void init_matrices();

    // In Weights : weights that are used to compute a neuron's value 
    // Out Weights : weights that are applied to a neuron to compute the next layer's neurons

    // It is through these methods that the *_weights matrices must be read and written to
    // They target a neuron on a particular layer of the neural network
    // 'layer' must be between 0 and nb_hidden_layer+2 and 'neuron' must be an valid index for the targeted layer
    // If the network does not have hidden layers then 'layer' must be between 0 and 1
    /* The {get|set}InWeights(...) methods deals with vectors containing the weights ( in order ) 
     * used to compute the value of the 'neuron'-th neuron on the 'layer'-th layer
     * The input layer's neurons do not have such weights, so 'layer' cannot be 0
     * The weights are stored as columns in the matrices
     */
    Eigen::VectorXd getInWeights( int layer, int neuron );
    void setInWeights( int layer, int neuron, Eigen::VectorXd& weights );
    /* The {get|set}OutWeights(...) methods deals with vectors containing the weights ( in order ) 
     * used on the 'neuron'-th neuron on the 'layer'-th layer to compute the neurons' value of the next layer
     * The output layer's neurons do not have such weights, so 'layer' cannot be nb_hidden_layers+2 
     * The weights are stored in the rows in the matrices
     */
    Eigen::VectorXd getOutWeights( int layer, int neuron );
    void setOutWeights( int layer, int neuron, Eigen::VectorXd& weights );

    double activation_function( bool is_used_for_classification, double value );

    /* Computes the value of every neuron in the network from a certain input 'input_k' and the weights stored in the *_weights matrices
     * The values of the neurons in hidden layers are stored in 'computed_hidden_neurons' whereas the values of the neurons of the output layer are stored in 'computed_output_neurons'
     * The computed_*_neurons matrices are initialized with the correct size before calling the method in train(...)
    */
    void compute_neuron_values( Eigen::VectorXd* neurons, bool is_used_for_classification );
    /* Corrects every weight of the neural network using the difference between the predicted output 'computed_output_neurons' and the correct output 'output_k' for the input 'input_k'
     * The computed_*_neurons matrices must be filled by a call to compute_neuron_values( 'input_k', ... ) 
     * alpha dictates the degree of change undergone by the weights every time they are updated
    */
    void update_weights( Eigen::VectorXd* neurons, Eigen::VectorXd& output_k, double alpha, bool is_used_for_classification );

    public : 
        MultiLayerPerceptron( int count, int* d )  
        {
            if( count < 2 )
            {
                // exception
            }

            nb_layers = count;
            nb_neurons_in_layer = new int[count];
            for( int i = 0; i < nb_layers; i++ )
                nb_neurons_in_layer[i] = d[i];

            init_matrices();
        }

        ~MultiLayerPerceptron()
        {
            delete nb_neurons_in_layer;
            delete weights;
        }

        /* Adds an element to the 'input' and 'output' matrices, which may be used to train the network
         * The first argument 'count' indicates the number of arguments that follows it
         * 'count' must be equal to [nb_neurons_in_input_layer+nb_neurons_in_output_layer]
         * Let X be an input and Y be an output associated to X. Then addElement is called like : 
         * AddElement( [X.size()+Y.size()], X[0], X[1], ..., X[X.size()-1], Y[0], Y[1], ..., Y[Y.size()-1] );
        */
        void addElement( int count, ... );
        void addElementArray( int count, double* array );
        /* Prints the content of the matrices
         * used purely for debugging
        */
        void printElements();

        void quickTrain();

        /* Trains the network for 'nb_iterations' iterations where an iteration consists of : 
         * 1) Choosing a random couple ( input[k], output[k] )
         * 2) Updating the weights of the network by calling compute_neuron_values(...) then update_weights(...)
         * If MSE_interval is greater than 0, the MSEvalues vector will get filled 
        */
        void train( int nb_iterations, double alpha, bool is_used_for_classification, int MSE_interval = 0 );

        /* Fills the member 'predictedOutput' by predicting the class in which the input given is
         * The first argument 'count' indicates the number of arguments that follows it
         * 'count' must be equal to [nb_neurons_in_input_layer]
         * Let X be an input. Then generatePrediction is called like : 
         * generatePrediction( X.size(), X[0], X[1], ..., X[X.size()-1] );
        */
        void generatePrediction( bool is_used_for_classification, int count, ... );
        void generatePredictionArray( bool is_used_for_classification, int count, double* array );
        /* Returns the element at the index 'index' in the vector 'predictedOutput'
        */
        double getPrediction( int index );

        int getMSESize() { return MSE_values.size(); }
        double MSE( int index ) { return MSE_values[index]; }
};

#endif