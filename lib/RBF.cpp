#include "RBF.hpp"

RBF::RBF( int init_nb_components_of_element, float init_gamma ) 
{
    nb_components_of_element = init_nb_components_of_element;
    gamma = init_gamma;
}

void RBF::initElements( int nbElements )
{
    input = Eigen::MatrixXf( nbElements, nb_components_of_element );
    output = Eigen::VectorXf( nbElements );
    weights = Eigen::VectorXf( nbElements );

    for( int i = 0; i < nbElements; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights[i] = weight * 2.0 - 1.0;
    }

    nb_elements = 0;
}

void RBF::addElement( int count, ... )
{
    va_list args;
    va_start( args, count );

    for( int i = 0; i < nb_components_of_element; i++ )
        input(nb_elements, i) = va_arg( args, double );

    output[nb_elements] = va_arg( args, double );

    nb_elements++;

    va_end( args ); 
}

void RBF::addElementArray( float* array )
{
    for( int i = 0; i < nb_components_of_element; i++ )
        input(nb_elements, i) = array[i];

    output[nb_elements] = array[nb_components_of_element];

    nb_elements++;
}

void RBF::print()
{
    std::cout << "X = \n" << input << std::endl << std::endl;
    std::cout << "C = \n" << clusters << std::endl << std::endl;
    std::cout << "Y^T = \n" << output.transpose() << std::endl << std::endl;
    std::cout << "W^T = \n" << weights.transpose() << std::endl << std::endl;
}

float RBF::predict( int count, ... )
{
    va_list args;
    va_start( args, count );

    Eigen::VectorXf X_k( nb_components_of_element );

    for( int i = 0; i < nb_components_of_element; i++ )
        X_k[i] = va_arg( args, double );

    va_end( args ); 
    return predictVector(X_k);
}

float RBF::predictArray( float* array )
{
    Eigen::VectorXf X_k( nb_components_of_element );

    for( int i = 0; i < nb_components_of_element; i++ )
        X_k[i] = array[i];

    return predictVector(X_k);
}

float RBF::dist( Eigen::VectorXf& a, Eigen::VectorXf& b )
{
    return (a-b).norm();
}

Eigen::VectorXf RBF::getGaussVector( Eigen::VectorXf& X )
{
    Eigen::MatrixXf centers = ( nb_clusters > 0 ) ? ( clusters ) : ( input );
    Eigen::VectorXf gaussX( centers.rows() );
    for( int i = 0; i < centers.rows(); i++ )
    {
        Eigen::VectorXf curRow = centers.row(i);
        gaussX[i] = exp( -gamma * dist( X, curRow ) );
    }
    return gaussX;
}

float RBF::predictVector( Eigen::VectorXf& X_k )
{
    Eigen::MatrixXf result = weights.transpose() * getGaussVector( X_k );
    return is_used_for_classification ? tanh( result(0,0) ) : result(0,0);
}

void RBF::generateClusters( int init_nb_clusters, int nb_iterations )
{
    if( init_nb_clusters >= nb_elements )
        throw std::runtime_error(std::string("Le nombre de clusters est plus grand ou égal au nombre d'éléments ( " + std::to_string(init_nb_clusters) + "et " + std::to_string(nb_elements) + " donnés à " + __FUNCTION__ + "() )"));

    // Initialization
    nb_clusters = init_nb_clusters;
    clusters = Eigen::MatrixXf( nb_clusters, nb_components_of_element );

    // contains indices of the elements of a cluster
    std::vector<int>* clustersPopulation = new std::vector<int>[nb_clusters];
    
    // Choose clusters at random
    std::vector<int> initClusterHeads;
    for( int i = 0; i < nb_clusters;  )
    {
        int randomIndex = rand()%nb_elements;
        bool alreadyAssigned = false;
        for( int index : initClusterHeads )
            if( index == randomIndex )
            {
                alreadyAssigned = true;
                break;
            }
        
        if( !alreadyAssigned )
        {
            clusters.row(i) = input.row(randomIndex);
            initClusterHeads.push_back(randomIndex);
            i++;
        }
    }

    for( int i = 0; i < nb_iterations; i++ )
    {
        for( int j = 0; j < nb_elements; j++ )
        {
            Eigen::VectorXf curElement = input.row(j);
            int clusterIndex = 0; 
            Eigen::VectorXf curClusterHead = clusters.row(0);
            float minDist = dist( curElement, curClusterHead );

            for( int k = 1; k < nb_clusters; k++ )
            {
                curClusterHead = clusters.row(k);
                float curDist = dist( curElement, curClusterHead );
                if( curDist < minDist )
                {
                    clusterIndex = k;
                    minDist = curDist;
                }
            }

            clustersPopulation[clusterIndex].push_back( j );
        }

        for( int j = 0; j < nb_clusters; j++ )
        {
            clusters.row(j) = input.row(clustersPopulation[j][0]);

            for( int k = 1; k < clustersPopulation[j].size(); k++ )
                clusters.row(j) += input.row(clustersPopulation[j][k]);
            
            clusters.row(j) /= static_cast<float>(clustersPopulation[j].size());

            clustersPopulation[j].clear();
        }
    }

    delete[] clustersPopulation;

    weights = Eigen::VectorXf( nb_clusters );

    for( int i = 0; i < nb_clusters; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights[i] = weight * 2.0 - 1.0;
    }
}

void RBF::train()
{
    if( nb_clusters > 0 )
    {
        Eigen::MatrixXf phi( nb_elements, nb_clusters );

        for( int i = 0; i < nb_elements; i++ )
            for( int j = 0; j < nb_clusters; j++ )
            {
                Eigen::VectorXf vec1 = input.row(i);
                Eigen::VectorXf vec2 = clusters.row(j);
                phi( i, j ) = exp( -gamma * dist( vec1, vec2 ) );
            }

        weights = ( phi.transpose() * phi ).inverse() * phi.transpose() * output;
    }
    else 
    {
        Eigen::MatrixXf phi( nb_elements, nb_elements );

        for( int i = 0; i < nb_elements; i++ )
            for( int j = 0; j < nb_elements; j++ )
            {
                Eigen::VectorXf vec1 = input.row(i);
                Eigen::VectorXf vec2 = input.row(j);
                phi( i, j ) = exp( -gamma * dist( vec1, vec2 ) );
            }

        weights = phi.inverse() * output;
    }
}

float RBF::test()
{
    float success = 0;

    for( int k = 0; k < nb_elements; k++ )
    {
        Eigen::VectorXf curElement = input.row(k);
        float res = predictVector( curElement );

        if( ( res > 0 &&  output[k] > 0 ) || ( res < 0 &&  output[k] < 0 ) )
            success += 1.0;
    }

    return (success*100.0)/static_cast<float>(nb_elements);
}