#include "exemple_Rosenblatt.hpp"

void ExempleRosenblatt::populate_points_random( float begin_range, float end_range, bool is_whole )
{
    for( int i = 0; i < nb_points; i++ )
    {
        Point point;
        point.generate_random( begin_range, end_range, is_whole );
        points.push_back( point );
    }
}

void ExempleRosenblatt::generate_weights()
{
    weights = Eigen::VectorXd(nb_weights);

    for( int i = 0; i < nb_weights; i++ )
    {
        float weight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights[i] = weight * 2.0 - 1.0;
    }
}

void ExempleRosenblatt::init_classify( std::string class1Color, std::string class2Color )
{
    for( int i = 0; i < nb_points; i++ )
    {
        Point& point = points[i];
        Color color;
        float class_value;

        if( point.getY() >= a*point.getX() + b )
        {
            color.set(class1Color);
            class_value = 1.0;
        }
        else 
        {
            color.set(class2Color);
            class_value = -1.0;
        }

        colors.push_back( color );
        class_values.push_back( class_value );
    }
}

float ExempleRosenblatt::predict( Eigen::VectorXd& X_k_with_one )
{
    Eigen::MatrixXd X_prime = X_k_with_one;
    Eigen::MatrixXd W_prime = weights.transpose();
    Eigen::MatrixXd result = W_prime * X_prime;
    return result(0,0) < 0 ? -1 : 1;
}

void ExempleRosenblatt::train( int nb_iterations, int interval )
{
    if( interval > 0 )
    {
        MSE_interval = interval;
        MSEs.resize(0);
    }

    float MSE = 0.0;

    for( int i = 0; i < nb_iterations; i++ )
    {
        int k = rand()%nb_points;
        Eigen::VectorXd X_k_with_one(3);
        X_k_with_one[0] = 1;
        for( int j = 0; j < 2; j++ ) { X_k_with_one[j+1] = points[k][j]; }
        float Y_k = class_values[k];
        float g_X_k = predict(X_k_with_one);
        weights = weights + alpha * ( Y_k - g_X_k ) * X_k_with_one;
        MSE += ( Y_k - g_X_k ) * ( Y_k - g_X_k );
        if( MSE_interval > 0 && (i+1)%MSE_interval == 0 )
        {
            std::cout << MSE/static_cast<float>(MSE_interval) << "\n";
            MSEs.push_back( MSE/static_cast<float>(MSE_interval) );
            MSE = 0;
        }
    }
}

void ExempleRosenblatt::print_points( std::string pathToFile )
{
    std::ofstream file;
    file.open(pathToFile);

    for( int i = 0; i < points.size(); i++ )
    {
        file << points[i][0] << " " << points[i][1] << " ";
        file << colors[i].getR() << " " << colors[i].getG() << " " << colors[i].getB() << std::endl;
    }

    file.close();
}

void ExempleRosenblatt::print_background( std::string pathToFile, std::string class1BG, std::string class2BG )
{
    std::ofstream file;
    file.open(pathToFile);

    for( int x = 0; x <= 100; x++ )
    {
        for( int y = 0; y <= 100; y++ )
        {
            Point p( static_cast<float>(x)/100.0, static_cast<float>(y)/100.0 );
            Eigen::VectorXd p_with_one(3);
            p_with_one[0] = 1;
            for( int j = 0; j < 2; j++ ) { p_with_one[j+1] = p[j]; }
            float predicted_value = predict( p_with_one );
            Color c( predicted_value < 0 ? class2BG : class1BG );
            
            file << p[0] << " " << p[1] << " ";
            file << c.getR() << " " << c.getG() << " " << c.getB() << std::endl;
        }
    }

    file.close();
}

void ExempleRosenblatt::print_MSE( std::string pathToFile, bool print_index )
{
    std::ofstream file;
    file.open(pathToFile);

    for( int i = 0; i < MSEs.size(); i++ )
    {
        if( print_index ) { file << (i+1)*MSE_interval << " "; }
        file << MSEs[i] << std::endl;
    }

    file.close();
}