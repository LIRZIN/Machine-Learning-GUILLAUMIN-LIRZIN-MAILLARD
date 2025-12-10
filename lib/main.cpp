#include "LinearModel.hpp"
#include "MLP.hpp"
#include "RBF.hpp"

using namespace std;

int main()
{

    /*
    RBF rbf(2, 0.1);
    int nb_elements = 200;
    rbf.setUsedForClassification( true );
    rbf.initElements( nb_elements );

    for( int i = 0; i < nb_elements; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x3 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y1 = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;
        rbf.addElement( x1, x2, y1 );
    }

    rbf.generateClusters( 5, 100 );

    rbf.train();
    //rbf.print();

    std::cout << "success rate : " << rbf.test() << "%\n";
    */
    /*
    LinearModel lm(2);

    lm.setUsedForClassification(true);
    lm.initElements( 4 );
    lm.addElement( 1.0, 1.0, 1.0 );
    lm.addElement( 2.0, 3.0, -1.0 );
    lm.addElement( 3.0, 3.0, -1.0 );
    lm.addElement( 3.0, 3.0, -1.0 );
    
    lm.train( 100000, 0.01, 1 );

    std::cout << "success rate : " << lm.test() << "%\n";

    lm.print(false, false, false, true);
    */
    /*
    int layers[2] = { 2, 1 };
    MLP mlp(2, layers);

    mlp.setUsedForClassification( true );
    mlp.initElements( 3 );
    mlp.addElement( 3, 1.0, 1.0, 1.0 );
    mlp.addElement( 3, 2.0, 3.0, -1.0 );
    mlp.addElement( 3, 3.0, 3.0, -1.0 );
    
    mlp.train( 100000, 0.01 );

    std::cout << "success rate : " << mlp.test() << "%\n";
    */

    int layers[3] = { 2, 3, 1 };
    MLP mlp(3, layers);
    mlp.setUsedForClassification(true);

    mlp.initElements( 4 );
    float test1[3] = {0,0,-1};
    float test2[3] = {1,0,1};
    float test3[3] = {0,1,1};
    float test4[3] = {1,1,-1};
    mlp.addElementArray(test1);
    mlp.addElementArray(test2);
    mlp.addElementArray(test3);
    mlp.addElementArray(test4);
    /*
    mlp.addElement( 3, 0, 0, -1 );
    mlp.addElement( 3, 1, 0, 1 );
    mlp.addElement( 3, 0, 1, 1 );
    mlp.addElement( 3, 1, 1, -1 );
    */

    mlp.train( 1000000, 0.01, 10000 );

    float test5[2] = {0.25,0};
    mlp.generatePredictionArray(test5);

    int test = mlp.getNbOutputNeurons();
    int result[1];
    for (int i = 0; i < mlp.getNbOutputNeurons(); i++) {
        result[i] = mlp.getPrediction(i) > 0 ? 1 : -1;
    }

    mlp.print(10);

    std::cout << "success rate : " << mlp.test() << "%\n";

    for (int i = 0; i < mlp.getMSESize(); i++) {

        std::cout << "MSE : " << mlp.MSE(i) << endl;
    }

    cout << "resultat : " << endl;
    for (int i = 0; i < mlp.getNbOutputNeurons(); i++) {
        cout << result[i] << " " << endl;
    }

    /*
    MLP mlp(2, 2);
    int nb_elements = 200;
    mlp.setUsedForClassification( true );
    mlp.initElements( nb_elements );

    for( int i = 0; i < nb_elements; i++ )
    {
        double x1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double x2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        double y1 = ( x2 >= -1.0*x1 + 0.7 ) ? 1.0 : -1.0;
        double y2 = ( x2 >= -1.0*x1 + 0.7 ) ? -1.0 : 1.0;
        mlp.addElement( 4, x1, x2, y1, y2 );
    }
    
    mlp.train( 100000, 0.01, 100 );

    std::cout << "success rate : " << mlp.test() << "%\n";
    */
}